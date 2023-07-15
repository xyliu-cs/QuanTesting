import torch
import torch.quantization
import torch.nn as nn
import requests
import time
from transformers import MBartForConditionalGeneration, MBart50TokenizerFast
from simalign import SentenceAligner
from revChatGPT.V1 import Chatbot
import numpy as np

model = None
tokenizer = None
quantized_model = None
myaligner = None
chatbot = None
file_path = './text.txt'
text = []
out =[]

HF_API_URL = "https://api-inference.huggingface.co/models/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
headers = {"Authorization": "your_authorization_code"}
access_token = "your_access_token"

stop_words = [
    'a', 'an', 'the', 'and', 'or', 'but', 'if', 'while', 'as', 'that', 'this',
    'these', 'those', 'to', 'for', 'with', 'at', 'from', 'by', 'on', 'off', 'of',
    'into', 'over', 'under', 'above', 'below', 'is', 'be', 'am', 'are', 'was',
    'were', 'been', 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'can',
    'could', 'shall', 'should', 'will', 'would', 'might', 'must', 'it', 'its',
    'it\'s', 'he', 'his', 'she', 'her', 'hers', 'they', 'their', 'theirs', 'you',
    'your', 'yours', 'we', 'our', 'ours', 'in', 'out', 'through', 'because',
    'while', 'during', 'before', 'after', 'about', 'against', 'between', 'among',
    'into', 'through', 'during', 'before', 'after', 'above', 'below', 'to',
    'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
    'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why', 'how',
    'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other', 'some', 'such',
    'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 's'
]

# helper functions
def mask_tokens(hit_map, tokens):
    non_punctuation_indices = [i for i, token in enumerate(tokens) if token.isalnum()]
    weights = [8 if hit_map[i] == 0 else 1/(hit_map[i]+1) for i in non_punctuation_indices]
    normalized_weights = [w / sum(weights) for w in weights]

    num_unref = sum(1 for i in non_punctuation_indices if hit_map[i] == 0)
    num_mask = np.random.randint(max(1, num_unref // 10), 6)
    mask_indices = np.random.choice(non_punctuation_indices, size=num_mask, p=normalized_weights, replace=False)

    # Continue generating mask_indices until we find one that isn't a stop word
    while all(tokens[i] in stop_words for i in mask_indices):
        num_mask = np.random.randint(max(1, num_unref // 10), 6)
        mask_indices = np.random.choice(non_punctuation_indices, size=num_mask, p=normalized_weights, replace=False)

    masked_tokens = tokens.copy()
    for i in mask_indices:
        if masked_tokens[i] not in stop_words:
            masked_tokens[i] = '<fill>'
  
    return ' '.join(masked_tokens)

def query(payload):
    retries = 0
    max_retries = 10
    while retries < max_retries:
        try:
            response = requests.post(HF_API_URL, headers=headers, json=payload)
            response.raise_for_status()  # This will raise an exception for HTTP errors
            data = response.json()
            # Check if the returned value is a list consisting of two numbers
            if isinstance(data, list) and len(data) == 2 and all(isinstance(i, (int, float)) for i in data):
                return data
            else:
                print(f'Returned value is not as expected, retrying... ({retries+1}/{max_retries})')
        except (requests.exceptions.RequestException, requests.exceptions.ConnectionError, ValueError) as e:
            print(f'Caught exception: {str(e)}, retrying... ({retries+1}/{max_retries})')
        retries += 1
        time.sleep(5)
    print("Payload is: ", payload)
    raise Exception('Max retries exceeded')

def process_tokens(token_list):
    processed_list = []
    for token in token_list:
        if token.startswith('▁'):  # remove the leading underscore
            token = token[1:]
        processed_list.append(token)
    return processed_list

def compare_tokens(original, quantized):
    original_indices = {token: [i for i, x in enumerate(original) if x == token] for token in original}
    quantized_indices = {token: [i for i, x in enumerate(quantized) if x == token] for token in quantized}
    
    additional_tokens = {token: quantized_indices[token] for token in quantized_indices if token not in original_indices}
    missing_tokens = {token: original_indices[token] for token in original_indices if token not in quantized_indices}
    
    return additional_tokens, missing_tokens

# additional is quantized output token index map; missing is baseline output token index map.
def build_hotmap(bmap, qmap, add, miss, src_len):
    hotmap = {i: 0 for i in range(src_len)}
    for token in add:
        for src_index, trg_index in qmap:
            if trg_index in add[token]:
                hotmap[src_index] += 1
                
    for token in miss:
        for src_index, trg_index in bmap:
            if trg_index in miss[token]:
                hotmap[src_index] += 1
    return hotmap

def get_completion(mask):
    prefix = "Complete the sentence by filling in the missing information denoted as <fill>: \n"
    prompt = prefix + mask

    response = ""
    for data in chatbot.ask(prompt):
        response = data["message"]

    return response












def initialize():
    global model, tokenizer, quantized_model, text, myaligner, chatbot

    print("Initializing the script...")
    print("Loading facebook/mbart-large-50-many-to-many-mmt NMT model as default.")
    model = MBartForConditionalGeneration.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")

    print("Loading facebook/mbart-large-50-many-to-many-mmt Tokenizer (en-zh) as default.")
    tokenizer = MBart50TokenizerFast.from_pretrained("facebook/mbart-large-50-many-to-many-mmt")
    # translate Chinese to English
    tokenizer.src_lang = "en_XX"

    print("Perform Pytorch dynamic quantization to int.8 as default.")
    quantized_model = torch.quantization.quantize_dynamic(model, {nn.Linear}, dtype=torch.qint8)

    print("Loading text from text.txt file as default.")
    with open(file_path, 'r') as file:
        sentences = file.readlines()
    text = [sentence.strip() for sentence in sentences]

    print("Loading word alignment model from simalign as default.")
    myaligner = SentenceAligner(model="bert", token_type="bpe", matching_methods="mai")

    print("Accessing the completion model using ChatGPT as default.")
    chatbot = Chatbot(config={"access_token": access_token})
    



def main():
    global out
    print("Running the main functionality...")
    suspect_ids = []
    normal_ids = []
    tokens = []

    ctr = 0
    while ctr < len(text):
        # just iterate 10 times for testing purpose
        if ctr > 10:
            break

        src_sent = text[ctr]
        encoded_en = tokenizer(src_sent, return_tensors="pt")
        input_tokens = [token for token in tokenizer.convert_ids_to_tokens(encoded_en['input_ids'][0]) if token not in ['<s>', '</s>']]
        generated_tokens = model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"], max_new_tokens=200)
        output_tokens = [token for token in tokenizer.convert_ids_to_tokens(generated_tokens[0]) if token not in ['<s>', '</s>']]
        generated_tokens_q = quantized_model.generate(**encoded_en, forced_bos_token_id=tokenizer.lang_code_to_id["zh_CN"], max_new_tokens=200)
        output_tokens_q = [token for token in tokenizer.convert_ids_to_tokens(generated_tokens_q[0]) if token not in ['<s>', '</s>']]
        tokens.append([input_tokens, output_tokens, output_tokens_q])

        decoded_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)
        decoded_text_q = tokenizer.batch_decode(generated_tokens_q, skip_special_tokens=True)
        out.append([src_sent, decoded_text[0], decoded_text_q[0]])
    
        base = decoded_text[0]
        quant = decoded_text_q[0]

        payload = {"inputs": {
                    "source_sentence": src_sent,
                    "sentences": [base, quant]}
                    }
        
        sim = query(payload)
        assert len(sim) == 2

        # Anomaly
        if (abs(sim[0] - sim[1]) >= 0.02) or (sim[1] <= 0.8):
            print("\nPotential translation error with similarity [base, quant] = ", sim)
            print("Source: ", src_sent)
            print("Baseline translation: ", base)
            print("Quantized translation: ", quant)
            suspect_ids.append(ctr)

            input_tokens = process_tokens(input_tokens[1:])
            output_tokens = process_tokens(output_tokens[1:])
            output_tokens_q = process_tokens(output_tokens_q[1:])

            addit, miss = compare_tokens(output_tokens, output_tokens_q)

            alignments = myaligner.get_word_aligns(input_tokens, output_tokens)
            alignments_q = myaligner.get_word_aligns(input_tokens, output_tokens_q)

            src_len = len(input_tokens)
            base_align = alignments['mwmf']
            quant_align = alignments_q['mwmf']

            hot_map = build_hotmap(base_align, quant_align, addit, miss, src_len)

            # maybe we should mask multiple times ???
            masked = mask_tokens(hot_map, input_tokens)

            # Should append to `text` and use a big while loop to iterate, now just print it out for testing purpose
            mutated_sent = get_completion(masked)

            print("\nOriginal: ", src_sent)
            print("Masked: ", masked)
            print("Mutated: ", mutated_sent)
            print("Appending mutated sentence to the sentence pool.")

            # the length of text should be increased by 1
            text.append(mutated_sent)
        else:
            normal_ids.append(ctr)
        
        ctr += 1



if __name__ == "__main__":
    initialize()
    main()





