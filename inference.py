from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torch
from datetime import datetime
from pytz import timezone
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

import argparse
parser = argparse.ArgumentParser(description="infer")
parser.add_argument("--data_path", "-data_path", type=str, help="저장된 데이터셋 경로")
args = parser.parse_args()

MAX_PATCHES = 512

def eval(dataset, deplot_model_path, t5_model_path, device):
    
    processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
    deplot_model = Pix2StructForConditionalGeneration.from_pretrained("ybelkada/pix2struct-base")
    deplot_model.load_state_dict(torch.load(deplot_model_path))
    
    tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-base")
    t5_model.load_state_dict(torch.load(t5_model_path))

    deplot_model.to(device)
    t5_model.to(device)
    
    deplot_model.eval()
    t5_model.eval()
    
    max_length = max([len(text) for text in dataset["description"]])

    f = open("/result/outputs/evaluation_log.txt", "a")
    current_time = datetime.now(timezone('Asia/Seoul'))
    print(f"Start: {current_time} Data: {len(dataset)}")
    f.write(f"Start: {current_time} Data: {len(dataset)}")
    
    data_list = []    
    for idx in tqdm(range(len(dataset))):
        data_id = dataset[idx]['data_id']
        image = dataset[idx]['image']
        datatable_label = dataset[idx]["text"]
        text_label = dataset[idx]["description"]
        
        inputs = processor(images=image, return_tensors="pt", max_patches=MAX_PATCHES).to(device)

        flattened_patches = inputs.flattened_patches
        attention_mask = inputs.attention_mask

        deplot_generated_ids = deplot_model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=1000)
        generated_datatable = processor.batch_decode(deplot_generated_ids, skip_special_token=True)[0]
        generated_datatable = generated_datatable.replace("<pad>", "").replace("<unk>", "").replace("</s>", "")
        
        tokenized_text = tokenizer.encode(generated_datatable, return_tensors='pt').to(device)
        
        t5_generated_ids = t5_model.generate(tokenized_text, max_length=max_length, num_beams=4, repetition_penalty=5.0, length_penalty=1.0, early_stopping=True, temperature=0.6)
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in t5_generated_ids]
        
        data_list.append([data_id, generated_datatable, datatable_label, preds, text_label])
    
    current_time = datetime.now(timezone('Asia/Seoul'))
    print(f"End: {current_time} Data: {len(dataset)}")
    f.write(f"End: {current_time} Data: {len(dataset)}")
    df = pd.DataFrame(data=data_list, columns=["data_id", "generated_table", "label_table", "generated_text", "label_text"])
    return df


if __name__ == '__main__':
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")

    dataset = load_dataset("imagefolder", data_dir=args.data_path, split="train")
    
    deplot_model_path = "model/deplot_k.pt"
    t5_model_path = "model/ke_t5.pt"
   
    result_df = eval(dataset, deplot_model_path, t5_model_path, device)
    result_df.to_csv("outputs/result.csv", index=False, header=True)    