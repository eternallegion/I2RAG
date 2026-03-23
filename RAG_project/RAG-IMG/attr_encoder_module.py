# attr_encoder_module.py

import sys
import torch
from typing import List, Dict

# CLEVR 인코더가 있는 경로를 sys.path에 추가 (경로는 환경에 맞게 수정)
sys.path.append("/mnt/c/Users/eternal/clevr_vit_package")

from clevr_vit import (
    CLEVRObjectLearner,
    GaussianMaskGenerator,
    preprocess_image,
)

# CLEVR 클래스 인덱스 → 이름 매핑 (데이터셋 기준으로 맞게 수정 가능)
COLOR_NAMES = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPE_NAMES = ["cube", "sphere", "cylinder"]
MATERIAL_NAMES = ["rubber", "metal"]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_clevr_encoder(ckpt_path: str | None = None) -> CLEVRObjectLearner:
    """
    CLEVRObjectLearner 로드.
    ckpt_path가 있으면 파인튜닝 체크포인트를 로드하고,
    없으면 base CLIP 백본 + 랜덤 head로 사용.
    """
    print("Initializing CLEVRObjectLearner (Base: openai/clip-vit-large-patch14-336)...")
    model = CLEVRObjectLearner(load_base=True).to(DEVICE)
    model.eval()

    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=DEVICE)
        # 체크포인트 포맷에 따라 key 이름이 다를 수 있음. 필요하면 여기 맞춰 조정.
        if "model" in state:
            model.load_state_dict(state["model"], strict=False)
        else:
            model.load_state_dict(state, strict=False)
        print(f"[attr_encoder] Loaded checkpoint from {ckpt_path}")
    else:
        print("[attr_encoder] Using base weights (no finetuned checkpoint)")

    return model


def infer_attributes_for_objects(
    model: CLEVRObjectLearner,
    image_path: str,
    objects: List[Dict],
) -> str:
    """
    image_path: CLEVR 스타일 이미지 경로
    objects: [{"pixel_coords": [x,y,z], "size": "small"/"large"}, ...]
    반환: 객체들에 대한 자연어 설명 텍스트 (RAG에 넣을 수 있음)
    """
    # 1) 이미지 전처리
    pixel_values = preprocess_image(image_path).to(DEVICE)

    # 2) Gaussian 마스크 생성
    mask_gen = GaussianMaskGenerator()
    masks = []
    for obj in objects:
        px, py, pz = obj["pixel_coords"]
        size = obj.get("size", "small")
        mask = mask_gen.generate_mask([px, py, pz], size_cat=size)  # [24,24]
        masks.append(mask)

    masks = torch.stack(masks, dim=0).unsqueeze(0).to(DEVICE)  # [1, N, 24, 24]

    # 3) 모델 추론
    with torch.no_grad():
        out = model(pixel_values, masks=masks)

    color_logits = out["color_logits"][0]       # [N, 8]
    shape_logits = out["shape_logits"][0]       # [N, 3]
    material_logits = out["material_logits"][0] # [N, 2]
    coords_pred = out["coords_pred"][0]         # [N, 3]

    # 4) 인덱스 → 텍스트 변환
    desc_lines = []
    N = color_logits.size(0)
    for i in range(N):
        color_idx = color_logits[i].argmax().item()
        shape_idx = shape_logits[i].argmax().item()
        material_idx = material_logits[i].argmax().item()
        x, y, z = coords_pred[i].tolist()

        color = COLOR_NAMES[color_idx]
        shape = SHAPE_NAMES[shape_idx]
        material = MATERIAL_NAMES[material_idx]

        line = (
            f"Object {i+1}: {color} {material} {shape} "
            f"located at (x={x:.2f}, y={y:.2f}, z={z:.2f})."
        )
        desc_lines.append(line)

    return "\n".join(desc_lines)
