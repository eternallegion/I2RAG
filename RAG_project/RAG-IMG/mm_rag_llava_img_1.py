# mm_rag_llava_img.py

import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# --- 설정 ---
VECTOR_DB_DIR = "./vector_db"
LLAVA_ID = "llava-hf/llava-1.5-7b-hf"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


def load_vectorstore():
    print(f"[vector_db] {VECTOR_DB_DIR} 에서 벡터 DB 로드 시도")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(
            f"{VECTOR_DB_DIR} 폴더가 없습니다. 먼저 build_vector_db_attr.py 를 실행해서 인덱스를 생성하세요."
        )

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("[vector_db] 로드 완료")
    return vectorstore


def load_llava():
    print(f"[LLaVA] Loading {LLAVA_ID} on {DEVICE} (dtype={DTYPE}) ...")
    model = LlavaForConditionalGeneration.from_pretrained(
        LLAVA_ID,
        dtype=DTYPE,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(LLAVA_ID)
    model.eval()
    print("[LLaVA] 로드 완료")
    return model, processor


def build_prompt(question: str, context_texts: List[str], use_image: bool) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{t}" for i, t in enumerate(context_texts)]
    )

    if use_image:
        intro = (
            "USER: <image>\n"
            "위 이미지는 단순한 도형이거나 상징적인 그림일 수 있습니다.\n"
            "이미지에서 보이는 내용은 있는 그대로만 묘사하고, 과도한 의미 부여는 피하세요.\n"
            "그 다음, 아래 문서 컨텍스트를 중심으로 질문에 답변해 주세요.\n\n"
        )
    else:
        intro = (
            "USER:\n"
            "아래에는 관련 문서에서 검색된 내용이 있습니다.\n"
            "문서 내용을 참고해서 질문에 답변해 주세요.\n\n"
        )

    prompt = (
        intro +
        f"{context_block}\n\n"
        f"질문: {question}\n"
        "1) 먼저 이미지를 간단히 묘사하고(이미지가 있는 경우),\n"
        "2) 그 다음 문서 내용을 기반으로 시스템 구조를 정리해서 한국어로 설명해줘.\n"
        "ASSISTANT:"
    )
    return prompt


def rag_llava_answer(
    question: str,
    retriever,
    llava_model,
    llava_processor,
    image_path: Optional[str] = None,
    top_k: int = 3,
    max_new_tokens: int = 512,
) -> str:
    # (1) RAG 검색
    docs = retriever.invoke(question)
    context_texts = [d.page_content for d in docs[:top_k]]

    use_image = image_path is not None

    # (2) 프롬프트 생성
    prompt = build_prompt(question, context_texts, use_image=use_image)

    # (3) 이미지 로드 (옵션)
    if use_image:
        image = Image.open(image_path).convert("RGB")
        inputs = llava_processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        ).to(DEVICE)
    else:
        inputs = llava_processor(
            text=prompt,
            images=None,
            return_tensors="pt",
        ).to(DEVICE)

    # (4) LLaVA 생성
    with torch.no_grad():
        out_ids = llava_model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    out_text = llava_processor.batch_decode(
        out_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    if "ASSISTANT:" in out_text:
        answer = out_text.split("ASSISTANT:")[-1].strip()
    else:
        answer = out_text.strip()

    return answer


def run_demo(question1: str, question2: str, image_path: Optional[str] = None):
    # (1) 벡터DB + retriever
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # (2) LLaVA 로드
    llava_model, llava_processor = load_llava()

    # --- 텍스트-only ---
    ans1 = rag_llava_answer(
        question=question1,
        retriever=retriever,
        llava_model=llava_model,
        llava_processor=llava_processor,
        image_path=None,
        max_new_tokens=512,
    )
    print("\n[질문1 - 텍스트만]\n", question1)
    print("\n[답변1]\n", ans1)

    # --- 이미지 포함 ---
    if image_path is not None and os.path.exists(image_path):
        ans2 = rag_llava_answer(
            question=question2,
            retriever=retriever,
            llava_model=llava_model,
            llava_processor=llava_processor,
            image_path=image_path,
            max_new_tokens=512,
        )
        print("\n[질문2 - 이미지 포함]\n", question2)
        print("[사용 이미지]", image_path)
        print("\n[답변2]\n", ans2)
    else:
        print(f"\n[info] 이미지가 없어서 이미지 포함 질문은 생략합니다: {image_path}")


if __name__ == "__main__":
    q1 = "이 문서들에서 설명하는 시스템의 전체 구조를 요약해줘."
    q2 = "이 이미지와 문서 내용을 함께 참고해서, 시스템의 구조를 설명해줘."
    run_demo(q1, q2, image_path="./test_image.png")
