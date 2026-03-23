# attr_encoder_integration.py

import torch
from clevr_vit import CLEVRObjectLearner, GaussianMaskGenerator, preprocess_image

# CLEVR 클래스 인덱스 → 문자열 매핑 (실제 CLEVR 기준 예시)
COLOR_NAMES = ["gray", "red", "blue", "green", "brown", "purple", "cyan", "yellow"]
SHAPE_NAMES = ["cube", "sphere", "cylinder"]
MATERIAL_NAMES = ["rubber", "metal"]

device = "cuda" if torch.cuda.is_available() else "cpu"

def load_clevr_encoder(ckpt_path: str | None = None):
    model = CLEVRObjectLearner(load_base=True).to(device)
    model.eval()
    if ckpt_path is not None:
        state = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state["model"], strict=False)  # 체크포인트 구조에 맞게 조정
        print(f"[attr-encoder] loaded checkpoint from {ckpt_path}")
    else:
        print("[attr-encoder] using base (unfined-tuned) weights")
    return model


def infer_attributes_for_objects(model, image_path: str, objects_json: list[dict]) -> str:
    """
    objects_json 예:
    [
      {"pixel_coords":[x,y,z], "size":"small"},
      {"pixel_coords":[x,y,z], "size":"large"},
      ...
    ]
    """
    # 1) 이미지 전처리
    pixel_values = preprocess_image(image_path).to(device)  # [1,3,336,336]

    # 2) 객체 마스크 생성
    mask_gen = GaussianMaskGenerator()
    masks = []
    for obj in objects_json:
        px, py, pz = obj["pixel_coords"]
        size = obj.get("size", "small")
        mask = mask_gen.generate_mask([px, py, pz], size_cat=size)  # [24,24]
        masks.append(mask)
    masks = torch.stack(masks, dim=0).unsqueeze(0).to(device)  # [1,N,24,24]

    # 3) 모델 추론
    with torch.no_grad():
        out = model(pixel_values, masks=masks)

    color_logits = out["color_logits"][0]      # [N,8]
    shape_logits = out["shape_logits"][0]      # [N,3]
    material_logits = out["material_logits"][0]# [N,2]
    coords_pred = out["coords_pred"][0]        # [N,3]

    # 4) argmax → 텍스트로 변환
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
