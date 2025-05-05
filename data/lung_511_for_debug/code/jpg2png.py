# from PIL import Image
# import os

# image_folder = 'blender_type/images'

# for filename in os.listdir(image_folder):
#     if filename.lower().endswith(('.jpg', '.jpeg')):
#         path_jpg = os.path.join(image_folder, filename)
#         path_png = os.path.join(image_folder, os.path.splitext(filename)[0] + '.png')

#         with Image.open(path_jpg) as img:
#             img = img.convert('RGBA')  # ⭐ 添加这一行
#             img.save(path_png)

#         print(f"Converted: {filename} -> {path_png}")


from PIL import Image

# 打开图片
image = Image.open("/home/data1/difanrui/Project/instant-nsr-pl/data/nerf_synthetic/hotdog/train/r_0.png")

# 获取图片的模式（例如 RGB, RGBA, L 等）
print("Image mode:", image.mode)

# 获取图片的通道数
if image.mode == "RGB":
    print("Number of channels: 3")
elif image.mode == "RGBA":
    print("Number of channels: 4")
elif image.mode == "L":
    print("Number of channels: 1")
else:
    print("Unknown number of channels")
