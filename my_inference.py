from datasets import load_dataset
from transformers import AutoProcessor, Pix2StructForConditionalGeneration, T5Tokenizer, T5ForConditionalGeneration
import torch
from datetime import datetime
from pytz import timezone
from tqdm import tqdm
import pandas as pd

import warnings
warnings.filterwarnings('ignore')

#데이터 경로 입력
import argparse
parser = argparse.ArgumentParser(description="infer")
parser.add_argument("--data_path", "-data_path", type=str, help="저장된 데이터셋 경로")
args = parser.parse_args()

MAX_PATCHES = 512

#모델 평가 함수 정의
def eval(dataset, deplot_model_path, t5_model_path, device):
    
    # DEPLOT 모델과 Processor 로딩
    processor = AutoProcessor.from_pretrained("ybelkada/pix2struct-base")
    deplot_model = Pix2StructForConditionalGeneration.from_pretrained("ybelkada/pix2struct-base")
    deplot_model.load_state_dict(torch.load(deplot_model_path))  # DEPLOT 모델 체크포인트 로드
    
    # KE-T5 모델과 토크나이저 로딩
    tokenizer = T5Tokenizer.from_pretrained("KETI-AIR/ke-t5-base")
    t5_model = T5ForConditionalGeneration.from_pretrained("KETI-AIR/ke-t5-base")
    t5_model.load_state_dict(torch.load(t5_model_path))  # KE-T5 모델 체크포인트 로드

    # 모델을 GPU 또는 CPU로 이동
    deplot_model.to(device)
    t5_model.to(device)
    
    # 모델을 평가 모드로 전환
    deplot_model.eval()
    t5_model.eval()
    
    # 설명 텍스트의 최대 길이 설정
    max_length = max([len(text) for text in dataset["description"]])

    # 평가 로그 파일 열기
    f = open("/result/outputs/evaluation_log.txt", "a")
    current_time = datetime.now(timezone('Asia/Seoul'))
    print(f"Start: {current_time} Data: {len(dataset)}")
    f.write(f"Start: {current_time} Data: {len(dataset)}")
    
    data_list = []  # 결과 데이터를 저장할 리스트
    
    # 데이터셋을 반복 처리
    for idx in tqdm(range(len(dataset))):
        data_id = dataset[idx]['data_id']  # 데이터 ID
        image = dataset[idx]['image']  # 이미지 데이터
        datatable_label = dataset[idx]["text"]  # 실제 라벨 (표 데이터)
        text_label = dataset[idx]["description"]  # 실제 설명 텍스트

        # 이미지를 전처리하여 모델에 입력
        inputs = processor(images=image, return_tensors="pt", max_patches=MAX_PATCHES).to(device)
        flattened_patches = inputs.flattened_patches
        attention_mask = inputs.attention_mask

        # DEPLOT 모델을 사용해 이미지에서 표 데이터 생성
        deplot_generated_ids = deplot_model.generate(flattened_patches=flattened_patches, attention_mask=attention_mask, max_length=1000)
        generated_datatable = processor.batch_decode(deplot_generated_ids, skip_special_token=True)[0]
        generated_datatable = generated_datatable.replace("<pad>", "").replace("<unk>", "").replace("</s>", "")

        # KE-T5 모델을 사용하여 텍스트 생성
        tokenized_text = tokenizer.encode(generated_datatable, return_tensors='pt').to(device)
        t5_generated_ids = t5_model.generate(tokenized_text, max_length=max_length, num_beams=4, repetition_penalty=5.0, length_penalty=1.0, early_stopping=True, temperature=0.6)
        preds = [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in t5_generated_ids]

        # 결과 저장
        data_list.append([data_id, generated_datatable, datatable_label, preds, text_label])
    
    # 평가 종료 시간 기록
    current_time = datetime.now(timezone('Asia/Seoul'))
    print(f"End: {current_time} Data: {len(dataset)}")
    f.write(f"End: {current_time} Data: {len(dataset)}")

    # 결과를 데이터프레임으로 변환하여 반환
    df = pd.DataFrame(data=data_list, columns=["data_id", "generated_table", "label_table", "generated_text", "label_text"])
    return df

#메인 함수 실행
if __name__ == '__main__':
    
    # 사용할 디바이스 결정 (GPU 또는 CPU)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device : {device}")

    # 데이터셋 로딩 (이미지 폴더에서 이미지 불러오기)
    dataset = load_dataset("imagefolder", data_dir=args.data_path, split="train")
    
    # 모델 경로 설정
    deplot_model_path = "../model/deplot_k.pt"
    t5_model_path = "../model/ke_t5.pt"
   
    # eval 함수 호출하여 추론 실행
    result_df = eval(dataset, deplot_model_path, t5_model_path, device)
    
    # 결과를 CSV 파일로 저장
    result_df.to_csv("../outputs/result.csv", index=False, header=True)  