from transformers import ElectraForSequenceClassification
import torch
from transformers import ElectraTokenizer
import argparse

from dotenv import load_dotenv
import os

import torch.nn as nn
import torch.backends.cudnn as cudnn
import random
import numpy as np

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')

# seed 고정
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.cuda.manual_seed_all(0)
np.random.seed(0)
cudnn.benchmark = False
cudnn.deterministic = True
random.seed(0)

try:
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print("모델을 로드했습니다.")
    # 학습된 모델을 로드
    model = ElectraForSequenceClassification.from_pretrained(MODEL_PATH)
    # 모델을 cpu에 할당
    model.to(device).eval()
except:
    raise AttributeError() 

def run(message):
    filter=""
    
    def get_parameters(message):
        parser = argparse.ArgumentParser(description = 'parameters for predict user input.')
        parser.add_argument('--input_text', help='user input text.', default=message, type=str)
        parser.add_argument('--base_ckpt', help='base path that saved trained checkpoints.', default=MODEL_PATH, type=str)
        args = parser.parse_args([])  # 빈 리스트를 전달하여 명령행 인자를 파싱하지 않도록 설정

        return args

    #매개변수로 받은 메세지와, 모델이 있는 경로 저장
    Args = get_parameters(message)

    tokenizer = ElectraTokenizer.from_pretrained(MODEL_PATH)

    def predict(toked_input, model):
        #입력받은 문장을 모델에 전달합니다.
        input_ids = toked_input['input_ids'].to(device)
        attention_mask = toked_input['attention_mask'].to(device)

        output = model(input_ids, attention_mask = attention_mask)[0]
        result = nn.Softmax(dim = 1)(output)

        return result

    label_list = ['bad', 'ok']
    
    tokenize_output = tokenizer.encode_plus(Args.input_text, max_length = 128, truncation=True, padding = 'max_length', return_tensors='pt')
        
    result = predict(tokenize_output, model)
    easy_result = np.argmax(result.data.cpu(), axis=1)
    
    # 메시지 값
    # print(Args.input_text)
    # 이진 분류 결과 값
    # print(result)
    # 나쁜말 판단 결과 값
    # print(label_list[int(easy_result.item())])
    filter=label_list[int(easy_result.item())]
        
    return filter