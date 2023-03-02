from transformers import AutoConfig, AutoTokenizer, HfArgumentParser, AutoModelForSeq2SeqLM, Trainer, GPT2LMHeadModel
import re
import json
from tqdm import tqdm

tokenizer = AutoTokenizer.from_pretrained("skt/kogpt2-base-v2", bos_token ='<start>', eos_token = '<s>', pad_token = '<pad>', unk_token = '<unk>', mask_token = '<mask>', sep_token ='<sep>')

model = GPT2LMHeadModel.from_pretrained("./gpt_generation_NER")
model.resize_token_embeddings(len(tokenizer))

ner_text = open("./result.txt", "w", encoding="utf-8")

with open("./Process_sample", "r") as f:
    json_data = json.load(f)

    for data in tqdm(json_data["data"]):
        asp = []

        #tokens = tokenizer.tokenize(data['원인'],  return_tensors="pt", truncation=True)
        #tokens = tokenizer.tokenize(data['현상'],  return_tensors="pt", truncation=True)
        tokens = tokenizer.tokenize(data['개선'],  return_tensors="pt", truncation=True)
        sent = " ".join(tokens)
        sent = "<start> " + sent + " <sep>"
        aq = tokenizer.encode(sent, return_tensors='pt', truncation=True)
        greedy_output = model.generate(aq, num_beams=3, max_length = 256, 
                        pad_token_id=tokenizer.pad_token_id,
                        eos_token_id=tokenizer.eos_token_id,
                        bos_token_id=tokenizer.bos_token_id)
        asp.append(greedy_output[0])

        decodes = []
        for a in asp:
            decodes.append(tokenizer.decode(a))

        result = str(decodes[0])

        p = re.compile('(?<=\[ )(.*?)(?= \])')
        tag_list = p.findall(result)

        new_tag_list = []
        for aa in tag_list:
            aa = aa.replace(" ", "")
            aa = aa.replace("|", ", ")
            new_tag_list.append(aa)

        ner_text.write(data['개선'] + "\n")
        ner_text.write((' ||  '.join(s for s in new_tag_list) + "\n\n"))
