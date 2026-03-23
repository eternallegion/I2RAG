# build_vector_db.py

import os
from config import DOC_DIR, VECTOR_DB_DIR, EMBEDDING_MODEL_NAME

from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


def load_documents(doc_dir: str):
    """DOC_DIR 안의 PDF / TXT를 LangChain Document 리스트로 로드."""
    docs = []
    for fname in os.listdir(doc_dir):
        path = os.path.join(doc_dir, fname)
        if fname.lower().endswith(".pdf"):
            print(f"[load] PDF: {fname}")
            loader = PyPDFLoader(path)
            docs.extend(loader.load())
        elif fname.lower().endswith(".txt"):
            print(f"[load] TXT: {fname}")
            loader = TextLoader(path, encoding="utf-8")
            docs.extend(loader.load())
        else:
            print(f"[skip] 지원하지 않는 형식: {fname}")
    print(f"[docs] 로드된 문서 수(청크 전): {len(docs)}")
    return docs


def build_vectorstore(docs):
    """문서를 청크로 자르고, FAISS 벡터스토어 생성."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)
    print(f"[docs] 청크된 문서 수: {len(splits)}")

    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},  # 노트북용 CPU 설정
        encode_kwargs={"normalize_embeddings": True},
    )

    vectorstore = FAISS.from_documents(splits, embeddings)
    print("[index] FAISS 벡터스토어 생성 완료")
    return vectorstore, embeddings


def main():
    print("[build_vector_db] 시작")

    docs = load_documents(DOC_DIR)
    if len(docs) == 0:
        print(f"[warn] {DOC_DIR} 에서 로드된 문서가 없습니다.")
        return

    vectorstore, _ = build_vectorstore(docs)

    # 디스크에 저장
    vectorstore.save_local(VECTOR_DB_DIR)
    print(f"[save] 벡터 DB 저장 완료: {VECTOR_DB_DIR}")


if __name__ == "__main__":
    main()
