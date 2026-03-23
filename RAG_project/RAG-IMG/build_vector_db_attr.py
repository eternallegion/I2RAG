# build_vector_db_attr.py

import os
from typing import List
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from attr_encoder_module import load_clevr_encoder, infer_attributes_for_objects

# --- 경로 및 설정 ---
DOC_DIR = "./docs"
VECTOR_DB_DIR = "./vector_db"
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_text_documents() -> List[Document]:
    docs: List[Document] = []

    if not os.path.exists(DOC_DIR):
        raise FileNotFoundError(f"{DOC_DIR} 폴더가 없습니다. 텍스트 문서를 먼저 준비해 주세요.")

    for fname in os.listdir(DOC_DIR):
        if not fname.lower().endswith(".txt"):
            continue
        fpath = os.path.join(DOC_DIR, fname)
        with open(fpath, "r", encoding="utf-8") as f:
            text = f.read()

        docs.append(Document(page_content=text, metadata={"source": fpath}))

    print(f"[build_vector_db_attr] 텍스트 문서 {len(docs)}개 로드 완료")
    return docs


def maybe_augment_with_attrs(docs: List[Document]) -> List[Document]:
    """
    예시:
    - docs[0]에 대해 test_image.png + 하드코딩된 objects 정보를 사용해
      CLEVR 인코더로 속성 텍스트를 생성하고 덧붙인다.
    실제 프로젝트에서는 문서별로 대응되는 이미지/객체 정보를 메타데이터로 매핑하는 구조로 확장하면 됨.
    """
    if not docs:
        return docs

    image_path = "./test_image.png"  # 실제 이미지 경로로 수정
    if not os.path.exists(image_path):
        print(f"[build_vector_db_attr] 이미지가 없어 속성 증강을 건너뜁니다: {image_path}")
        return docs

    # 예시용 객체 정보 (실제 CLEVR 메타데이터로 교체 가능)
    sample_objects = [
        {"pixel_coords": [120, 160, 4], "size": "small"},
        {"pixel_coords": [300, 200, 6], "size": "large"},
    ]

    print("[build_vector_db_attr] CLEVR 인코더 로드 및 속성 텍스트 생성...")
    clevr_model = load_clevr_encoder(ckpt_path=None)  # 필요하면 체크포인트 경로 지정

    attr_text = infer_attributes_for_objects(
        model=clevr_model,
        image_path=image_path,
        objects=sample_objects,
    )

    # 첫 번째 문서에 속성 텍스트를 단순히 덧붙이는 예시
    docs[0].page_content += "\n\n[Image Objects]\n" + attr_text
    print("[build_vector_db_attr] 첫 문서에 속성 텍스트를 추가했습니다.")

    return docs


def build_vector_db_with_attrs(force_rebuild: bool = False):
    # 이미 벡터 DB가 있고, 재빌드가 아니면 스킵
    if os.path.exists(VECTOR_DB_DIR) and not force_rebuild:
        print(f"[build_vector_db_attr] {VECTOR_DB_DIR} 가 이미 존재합니다. (force_rebuild=False 이므로 스킵)")
        return

    os.makedirs(VECTOR_DB_DIR, exist_ok=True)

    docs = load_text_documents()
    docs = maybe_augment_with_attrs(docs)

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )

    print("[build_vector_db_attr] FAISS 인덱스 생성 중...")
    vectorstore = FAISS.from_documents(docs, embeddings)

    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[build_vector_db_attr] 벡터 DB 생성 및 저장 완료: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    build_vector_db_with_attrs(force_rebuild=True)
