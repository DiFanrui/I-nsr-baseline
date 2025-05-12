from PIL import Image, ImageDraw, ImageFont
import os

def add_id_to_images(input_dir, output_dir, font_path="/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size=50):
    """
    在图片左上角添加文件名作为 ID，并保存到新的目录。

    Args:
        input_dir (str): 输入图片目录路径。
        output_dir (str): 输出图片目录路径。
        font_path (str): 字体文件路径（可选）。
        font_size (int): 字体大小。
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 加载字体
    if font_path and os.path.exists(font_path):
        font = ImageFont.truetype(font_path, font_size)
    else:
        font = ImageFont.load_default()

    # 遍历输入目录中的所有文件
    for file_name in os.listdir(input_dir):
        input_path = os.path.join(input_dir, file_name)
        if not os.path.isfile(input_path):
            continue

        try:
            # 打开图片
            img = Image.open(input_path)
            draw = ImageDraw.Draw(img)

            # 添加文件名到左上角
            text = os.path.splitext(file_name)[0]  # 去掉文件扩展名
            draw.text((10, 10), text, fill="black", font=font)

            # 保存到输出目录
            output_path = os.path.join(output_dir, file_name)
            img.save(output_path)
            print(f"Processed: {file_name}")
        except Exception as e:
            print(f"Failed to process {file_name}: {e}")

if __name__ == "__main__":
    input_dir = "exp/neus-blender/use_10_images_1to91stride10_only_render/save/it20000-test/images"  # 输入图片目录
    output_dir = "exp/neus-blender/use_10_images_1to91stride10_only_render/save/it20000-test_with_id/images"  # 输出图片目录
    # font_path = None  # 可选：指定字体路径，例如 "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
    # font_size = 300

    add_id_to_images(input_dir, output_dir)