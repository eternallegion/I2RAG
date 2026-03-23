# mm_rag_llava.py

import os
from typing import List, Optional

import torch
from PIL import Image
from transformers import AutoProcessor, LlavaForConditionalGeneration

from config import (
    VECTOR_DB_DIR,
    LLAVA_ID,
    DEVICE,
    DTYPE,
    EMBEDDING_MODEL_NAME,
)
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ============================
# 1. 벡터 DB 로드 (LangChain + FAISS)
# ============================

def load_vectorstore():
    print(f"[vector_db] {VECTOR_DB_DIR} 에서 벡터 DB 로드 시도")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    if not os.path.exists(VECTOR_DB_DIR):
        raise FileNotFoundError(
            f"{VECTOR_DB_DIR} 폴더가 없습니다. 먼저 build_vector_db.py 를 실행해서 인덱스를 생성하세요."
        )

    vectorstore = FAISS.load_local(
        VECTOR_DB_DIR,
        embeddings,
        allow_dangerous_deserialization=True,
    )
    print("[vector_db] 로드 완료")
    return vectorstore


# ============================
# 2. LLaVA 로드
# ============================

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


# ============================
# 3. 프롬프트 생성
# ============================

def build_prompt(question: str, context_texts: List[str]) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{t}" for i, t in enumerate(context_texts)]
    )

    prompt = (
        "USER:\n"
        "다음은 관련 문서에서 검색된 내용입니다.\n"
        f"{context_block}\n\n"
        f"질문: {question}\n"
        "위의 내용을 최대한 참고해서, 한국어로 자세히 설명해줘.\n"
        "ASSISTANT:"
    )
    return prompt


# ============================
# 4. RAG + LLaVA 인퍼런스
# ============================

def rag_llava_answer(
    question: str,
    retriever,
    llava_model,
    llava_processor,
    image_path: Optional[str] = None,
    top_k: int = 3,
    max_new_tokens: int = 1024,
) -> str:
    # (1) RAG 검색 (LangChain v0.3+ 스타일)
    docs = retriever.invoke(question)
    context_texts = [d.page_content for d in docs[:top_k]]

    # (2) 프롬프트 생성
    prompt = build_prompt(question, context_texts)

    # (3) 이미지 로드 (옵션)
    if image_path is not None:
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


# ============================
# 5. main: 예시 실행
# ============================

def main():
    # (1) 벡터DB 로드 + retriever 준비
    vectorstore = load_vectorstore()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # (2) LLaVA 로드 (4090 있는 PC에서 돌리는 걸 추천)
    llava_model, llava_processor = load_llava()

    # (3) 텍스트-only RAG + LLaVA 예시
    question = "이 문서들에서 설명하는 시스템의 전체 구조를 요약해줘."
    answer = rag_llava_answer(
        question=question,
        retriever=retriever,
        llava_model=llava_model,
        llava_processor=llava_processor,
        image_path=None,  # 이미지 없이
        max_new_tokens=1024,
    )

    print("\n[질문]\n", question)
    print("\n[답변]\n", answer)

    # (4) 멀티모달 예시 (이미지와 함께 사용하고 싶을 때)
    # image_path = "./images/diagram1.png"
    # if os.path.exists(image_path):
    #     q2 = "이 이미지와 문서 내용을 동시에 고려해서 시스템을 설명해줘."
    #     ans2 = rag_llava_answer(
    #         question=q2,
    #         retriever=retriever,
    #         llava_model=llava_model,
    #         llava_processor=llava_processor,
    #         image_path=image_path,
    #     )
    #     print("\n[질문2]\n", q2)
    #     print("\n[답변2]\n", ans2)


if __name__ == "__main__":
    main()
