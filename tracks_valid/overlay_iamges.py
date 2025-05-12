import cv2
import numpy as np
import matplotlib.pyplot as plt

def overlay_images(gt_image_path, render_image_path, alpha=0.5):
    """
    将两张图像重叠，其中一张图像设置为透明度 alpha。
    
    Args:
        gt_image_path (str): Ground Truth 图像路径。
        render_image_path (str): 渲染图像路径。
        alpha (float): 渲染图像的透明度，范围 [0, 1]。
    """
    # 加载两张图像
    gt_img = cv2.imread(gt_image_path, cv2.IMREAD_COLOR)  # 加载 GT 图像为 RGB
    render_img = cv2.imread(render_image_path, cv2.IMREAD_GRAYSCALE)  # 加载 Render 图像为灰度

    if gt_img is None or render_img is None:
        raise ValueError("Failed to load one or both images.")

    # 确保两张图像大小一致
    if gt_img.shape[:2] != render_img.shape:
        raise ValueError("The two images must have the same dimensions.")

    # 将灰度图像转换为彩色（3通道）
    render_colored = cv2.cvtColor(render_img, cv2.COLOR_GRAY2BGR)

    # 将两张图像叠加
    blended = cv2.addWeighted(gt_img, 1 - alpha, render_colored, alpha, 0)  # 叠加图像

    # 显示所有图像
    plt.figure(figsize=(12, 8))

    # 显示 GT 原图
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(gt_img, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供 Matplotlib 显示
    plt.title('GT Original (RGB)')
    plt.axis('off')

    # 显示 Render 原图
    plt.subplot(1, 3, 2)
    plt.imshow(render_img, cmap='gray')  # 直接显示灰度图
    plt.title('Render Original (Gray)')
    plt.axis('off')

    # 显示叠加结果
    plt.subplot(1, 3, 3)
    plt.imshow(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供 Matplotlib 显示
    plt.title('Overlayed Images')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

# 调用示例
overlay_images("data/lung_511_for_debug/images/00000001.png", "exp/neus-blender/use_3_images_1and11and51_only_render/save/it20000-test/00000001.png", alpha=0.5)