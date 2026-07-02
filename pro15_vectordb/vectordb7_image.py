# VectorDB에 Image 저장 후 검색 - 이미지로 이미지 검색
# pip install huggingface_hub

from huggingface_hub import list_models
models = list_models(search='clip', limit=20)
for m in models:
    print(m.modelId)


#######################################################################################
# clip model로 Image Embedding
# pip install torch pillow transformers
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import torch
import numpy as np
import matplotlib as plt
import koreanize_matplotlib
from chromadb import PersistentClient
from transformers import CLIPProcessor, CLIPModel
from numpy.linalg import norm
from PIL import image


#######################################################################################
# CLIP model 준비 작업
model_name = 'openai/clip-vit-base-patch32'     # Hugging Face에 등록된 CLIP 기본 모델
processor = CLIPProcessor.from_pretrained(model_name)   # 데이터를 CLIP 입력 형식으로 전처리
model = CLIPModel.from_pretrained(model_name)   # 데이터를 밀집 벡터로 변환

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model.to(device)    # 모델을 선택한 장치로 이동함
model.eval()    # 학습용 모델이 아니라 추론용으로 사용

print('모델 이름 : ', model_name)
print('사용 장치 : ', device)
print('모델 타입 : ', type(model))



#######################################################################################
# Image를 CLIP vector로 변환
def image_to_vector(img_path):
    image = image.open(img_path).convert('RGB')

    inputs = processor(     # Image를 CLIP model 입력 형식으로 변환
        images=image,       # 전처리 대상
        return_tensors='pt'     # 결과를 PyTorch 텐서 형식으로 반환
    )
    #### 수업 이어서