# orchestrator.py

import os

from build_vector_db_attr import build_vector_db_with_attrs
from mm_rag_llava_img_2 import run_demo

VECTOR_DB_DIR = "./vector_db"


def main():
    print("=== [1] 벡터 DB 상태 확인 ===")
    if not os.path.exists(VECTOR_DB_DIR):
        print(f"{VECTOR_DB_DIR} 가 없으므로, 새로 생성합니다.")
        build_vector_db_with_attrs(force_rebuild=True)
    else:
        print(f"{VECTOR_DB_DIR} 가 이미 존재합니다. 재빌드 없이 그대로 사용합니다.")

    print("\n=== [2] RAG + LLaVA 데모 실행 ===")
    question1 = "이 문서들에서 설명하는 시스템의 전체 구조를 요약해줘."
    question2 = "이 이미지와 문서 내용을 함께 참고해서, 시스템의 구조를 설명해줘."

    run_demo(
        question1,
        question2,
        image_path="./test_image.png",  # 실제 이미지 경로로 수정 가능
    )

    print("\n=== 파이프라인 완료 ===")


if __name__ == "__main__":
    main()
