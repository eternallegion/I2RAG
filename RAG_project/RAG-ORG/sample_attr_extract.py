# sample_attr_extract.py
import sys
sys.path.append("/mnt/c/Users/eternal/clevr_vit_package")

import json
import torch

from clevr_vit import (
    CLEVRObjectLearner,
    GaussianMaskGenerator,
    preprocess_image,
)

# ----- (1) CLEVR 인코더 로드 -----
device = "cuda" if torch.cuda.is_available() else "cpu"

model = CLEVRObjectLearner(load_base=True).to(device)
model.eval()

print("[INFO] CLEVR 인코더 로드 완료")


# ----- (2) 테스트용 이미지 & 객체 JSON -----

IMAGE_PATH = "test_image.png"           # 제공된 예시 이미지
JSON_PATH = "sample_objects.json"         # 객체 좌표 JSON

with open(JSON_PATH, "r") as f:
    objects = json.load(f)

print(f"[INFO] 객체 개수: {len(objects)}")


# ----- (3) 이미지 전처리 -----
pixel_values = preprocess_image(IMAGE_PATH).to(device)


# ----- (4) Gaussian 마스크 생성 -----
mask_gen = GaussianMaskGenerator()
masks = []

for obj in objects:
    px, py, pz = obj["pixel_coords"]
    size = obj.get("size", "small")
    mask = mask_gen.generate_mask([px, py, pz], size_cat=size)
    masks.append(mask)

masks = torch.stack(masks, dim=0).unsqueeze(0).to(device)  # [1, N, 24, 24]


# ----- (5) 모델 추론 -----
with torch.no_grad():
    out = model(pixel_values, masks=masks)

color_logits = out["color_logits"][0]      # [N,8]
shape_logits = out["shape_logits"][0]      # [N,3]
material_logits = out["material_logits"][0]# [N,2]
coords_pred = out["coords_pred"][0]        # [N,3]


# ----- (6) 클래스 매핑 -----
COLOR = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPE = ["cube", "sphere", "cylinder"]
MATERIAL = ["rubber", "metal"]


# ----- (7) 속성 → 자연어 텍스트 변환 -----

print("\n[결과] 객체 속성 텍스트\n")

for i in range(len(objects)):
    color = COLOR[color_logits[i].argmax()]
    shape = SHAPE[shape_logits[i].argmax()]
    material = MATERIAL[material_logits[i].argmax()]
    x, y, z = coords_pred[i].tolist()

    print(f"Object {i+1}:")
    print(f" - 색상: {color}")
    print(f" - 형태: {shape}")
    print(f" - 재질: {material}")
    print(f" - 좌표: (x={x:.2f}, y={y:.2f}, z={z:.2f})")
    print("----------------------------")
