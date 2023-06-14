from transformers import AutoTokenizer, AutoModelForSequenceClassification
from dotenv import load_dotenv
import os

load_dotenv()
MODEL_PATH = os.getenv('MODEL_PATH')
HUGGIN_FACE_PATH = os.getenv('HUGGIN_FACE_PATH')

tokenizer = AutoTokenizer.from_pretrained(HUGGIN_FACE_PATH)

# 모델을 가져옵니다.
model = AutoModelForSequenceClassification.from_pretrained(HUGGIN_FACE_PATH)

# 모델을 폴더에 저장
tokenizer.save_pretrained(MODEL_PATH)
model.save_pretrained(MODEL_PATH)