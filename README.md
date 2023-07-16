# QuanTesting
## Demo
![Demo](https://github.com/xyliu-cs/QuanTesting/blob/8d285118f8668cbcc10e58bf95dec39ecf7358be/dark.jpg)

## Theory
* **Theory 1**: From the paper *DiffChaser: Detecting Disagreements for Deep Neural Networks* [\[IJCAI'19\]](https://www.ijcai.org/proceedings/2019/0800.pdf), we know that the decision boundaries between a DNN and its quantized version variant are often quite **similar**.
* **Theory 2**: From the same paper, we know that the inputs near the decision boundary are more likely to capture the different behaviors of a DNN and its quantized version, falling into the so-called **disagreement region**.
* **Lemma 1**: From 1 and 2, we can infer that if an anomalous input within the disagreement region undergoes only slight mutation, it is plausible that the mutated input will remain within the confines of the disagreement region.

This work aims to automate the generation of inputs that provoke translation discrepancies or errors within the quantized Neural Machine Translation (NMT) model, thereby facilitating the evaluation of quantization robustness. **QuanTesting** treats sentences that induce detectable sentence-embedding differences for an NMT model and its quantized version as the anomalous inputs. For those inputs, it aligns and pinpoints the tokens responsible for the translation discrepancies and assigns higher masking probabilities to innocuous tokens and lower probabilities to critical tokens based on their ranked responsibilities. It then randomly masks some tokens according to this weighted distribution to create masked sentences. The masked sentence is subsequently fed into the completion model with a standard prompt to create a mutant. The mutant sentences are again appended to the sentence pool, waiting to be translated by the models. This cycle of translation, embedding comparison, alignment, masking, and mutation can be iterated to expand the set of detected anomalies.

## Sample workflow
| Iter 1 | Translation | 
| --------------- | --------------- | 
| Source    | According to a 2018 Harvard Business School study, venture capital firms that increased the number of female partners they hired by even 10% saw a bump in overall fund returns.|
| Baseline  | 根据哈佛商学院(Harvard Business School)的一份2018年研究,那些增加雇佣女性合作伙伴的创业投资公司( venture capital firms)的总回报率甚至增加了10%。|  
| Quantized | 哈佛商学院(Harvard Business School)2018年的一项研究显示,在增加雇佣的女性合伙人人数甚至为10%的风险资本企业,总资金回报大幅上升。 |  
| Masked    | According to a \<fill\> Harvard Business School study , venture capital firm s that increased the number of female partners they hi red by even 10% \<fill\> a bu mp in overall fund return s .|  
| Mutated   | According to a recent Harvard Business School study, venture capital firms that increased the number of female partners they hired by even 10% experienced a bump in overall fund returns.|  


| Iter 2 | Translation | 
| --------------- | --------------- | 
| Source    | According to a recent Harvard Business School study, venture capital firms that increased the number of female partners they hired by even 10% experienced a bump in overall fund returns.|  
| Baseline  | 根据哈佛商学院最近的一项研究,那些增加雇佣女性合伙人人数甚至为10%的风险资本公司,总体基金回报率大幅上升。|  
| Quantized | 根据哈佛商学院最近的一项研究,风险投资公司( venture capital firms)增加雇佣的女性合伙人人数甚至为10 %, 因此总资金回报大幅下滑。(new anomaly found) |  


To help understand, a reference translation is given as:
|           | Translation | 
| --------------- | --------------- | 
| Reference | 根据哈佛商学院最近的一项研究，风险投资公司聘用的女性合伙人数量哪怕增加10%，其基金的整体回报也会增加。|


## Usage
To use this work, the following conditions should be satisfied:
```bash
python 3.9.1+
revChatGPT 3.6.1 (access token)
PyTorch 1.9+
Transformers 4.0.0+
simalign
Hugging Face Inference API authentication code 
```


To run the script:
```bash
cd src
# replace text.txt with your own file
# replace the access tokens in quant_testing.py with your own tokens
python quant_testing.py
```
Alternatively, to experiment interactively:
```bash
jupyter notebook quant_testing_interactive.ipynb
```
