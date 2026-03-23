# config.py

import torch
import os

# 📂 문서가 들어 있는 폴더 (PDF / TXT 넣어두기)
DOC_DIR = "./docs"

# 📂 벡터 DB 저장/로드 위치
VECTOR_DB_DIR = "./vector_db"

# 🧠 텍스트 임베딩 모델 (LangChain용)
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# 🤖 LLaVA 모델 이름 (HF 허브)
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"

# 🔧 디바이스 / dtype 설정
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

os.makedirs(DOC_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

print(f"[config] DEVICE={DEVICE}, DTYPE={DTYPE}")
print(f"[config] DOC_DIR={DOC_DIR}")
print(f"[config] VECTOR_DB_DIR={VECTOR_DB_DIR}")
