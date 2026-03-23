# mm_rag_llava_img.py (개선 버전)

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

'''
def build_prompt(question: str, context_texts: List[str], use_image: bool) -> str:
    """
    RAG로 검색된 컨텍스트와 사용자의 질문을 LLaVA에 넘길 프롬프트를 생성.
    텍스트-only / 이미지+텍스트 상황을 명확히 분리해서 품질을 높인다.
    """
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{t}" for i, t in enumerate(context_texts)]
    )

    if use_image:
        # 이미지가 있는 경우: 이미지 설명은 짧게, 문서 기반 구조 설명이 메인
        prompt = f"""SYSTEM:
당신은 신중하고 정확한 한국어 기술 설명 어시스턴트입니다.
아래 규칙을 반드시 지키세요.
- 먼저, 이미지에서 눈에 보이는 것만 간단히 묘사합니다.
- 이미지를 과도하게 해석하거나 의미를 추측하지 않습니다.
- 단순 도형/배경만 있는 경우, "단순한 도형 혹은 배경만 있는 이미지"라고만 짧게 언급합니다.
- 그 다음, 주어진 문서 컨텍스트(Context)만을 근거로 시스템의 구조를 2~4개의 단락으로 논리적으로 설명합니다.
- 문서에 없는 내용은 절대 상상해서 만들지 말고, 정보가 부족하다고 명시합니다.

USER:
다음은 관련 문서에서 검색된 내용입니다.

{context_block}

위 문서들과 함께, 첨부된 이미지를 참고하여 아래 질문에 답변해 주세요.

질문: {question}

ASSISTANT:
"""
    else:
        # 텍스트만 있는 경우: 이미지 언급 금지
        prompt = f"""SYSTEM:
당신은 신중하고 정확한 한국어 기술 설명 어시스턴트입니다.
아래 규칙을 반드시 지키세요.
- 이미지는 주어지지 않았으므로, 이미지에 대해서는 어떤 언급도 하지 않습니다.
- 주어진 문서 컨텍스트(Context)만을 근거로 시스템의 구조를 2~4개의 단락으로 논리적으로 설명합니다.
- 문서에 없는 내용은 상상해서 만들지 말고, 정보가 부족하다고 명시합니다.

USER:
다음은 관련 문서에서 검색된 내용입니다.

{context_block}

위 문서들을 참고하여 아래 질문에 답변해 주세요.

질문: {question}

ASSISTANT:
"""           셈플이미지가 적어서 여기는 일단 봉인함
    return prompt
'''

def build_prompt(question: str, context_texts: List[str], use_image: bool) -> str:
    context_block = "\n\n".join(
        [f"[Context {i+1}]\n{t}" for i, t in enumerate(context_texts)]
    )

    if use_image:
        prompt = f"""SYSTEM:
당신은 신중하고 정확한 한국어 기술 설명 어시스턴트입니다.
아래 규칙을 반드시 지키세요.
- 먼저, 이미지에서 눈에 보이는 것만 1~2문장으로 아주 짧게 묘사합니다.
- 이미지를 과도하게 해석하거나 의미를 추측하지 않습니다.
- 단순 도형/배경만 있는 경우, "단순한 도형 혹은 배경만 있는 이미지"라고만 짧게 언급합니다.
- 그 다음에는 반드시 문서 컨텍스트(Context)에 집중하여 시스템의 구조를 설명해야 합니다.
- 문서 컨텍스트만으로도 설명 가능한 내용은, 이미지에 정보가 없더라도 끝까지 설명합니다.
- 답변은 번호 매기지 말고, 2~4개의 자연스러운 단락으로 작성합니다.
- 문서에 전혀 없는 내용은 상상해서 만들지 말고, 정보가 부족하다고 명시합니다.

USER: <image>
다음은 관련 문서에서 검색된 내용입니다.

{context_block}

위 문서들과 이미지를 함께 참고하여 아래 질문에 답변해 주세요.

질문: {question}

ASSISTANT:
"""
    else:
        prompt = f"""SYSTEM:
당신은 신중하고 정확한 한국어 기술 설명 어시스턴트입니다.
아래 규칙을 반드시 지키세요.
- 이미지는 주어지지 않았으므로, 이미지에 대해서는 어떤 언급도 하지 않습니다.
- 주어진 문서 컨텍스트(Context)에 기반하여 시스템의 구조를 2~4개의 단락으로 논리적으로 설명합니다.
- 번호 목록(1,2,3...) 대신 연속된 문단으로 서술합니다.
- 문서에 없는 내용은 상상해서 만들지 말고, 정보가 부족하다고 명시합니다.

USER:
다음은 관련 문서에서 검색된 내용입니다.

{context_block}

위 문서들을 참고하여 아래 질문에 답변해 주세요.

질문: {question}

ASSISTANT:
"""
    return prompt



def rag_llava_answer(
    question: str,
    retriever,
    llava_model,
    llava_processor,
    image_path: Optional[str] = None,
    top_k: int = 3,
    max_new_tokens: int = 320,
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
            do_sample=False,  # 결정적 출력
        )

    out_text = llava_processor.batch_decode(
        out_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]

    # LLaVA 출력에서 "ASSISTANT:" 뒷부분만 가져오기
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
        max_new_tokens=320,
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
            max_new_tokens=320,
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
