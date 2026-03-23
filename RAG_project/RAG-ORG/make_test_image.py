# make_test_image.py
from PIL import Image, ImageDraw

# 단순한 빨간 사각형 이미지
img = Image.new("RGB", (256, 256), "white")
draw = ImageDraw.Draw(img)
draw.rectangle([50, 50, 200, 200], outline="red", width=5)

img.save("test_image.png")
print("test_image.png 생성됨!")
