import cv2
import numpy as np
import matplotlib.pyplot as plt

def create_pattern_overlay(shape, mask, color, direction="right"):
    """
    创建带有斜线纹理的彩色图像，斜线与白色交织。
    
    Args:
        shape (tuple): 图像的形状 (height, width, channels)。
        mask (ndarray): 二值化的掩码图像。
        color (tuple): 颜色 (R, G, B)。
        direction (str): 斜线方向 ("right" 或 "left")。
    
    Returns:
        ndarray: 带有斜线纹理的彩色图像。
    """
    overlay = np.ones(shape, dtype=np.uint8) * 255  # 初始化为白色背景
    h, w = mask.shape

    # 创建斜线纹理
    for y in range(h):
        for x in range(w):
            if mask[y, x] > 0:  # 只在 mask 区域内绘制
                if direction == "right" and (x + y) % 10 < 5:  # 右下斜线
                    overlay[y, x] = color
                elif direction == "left" and (x - y) % 10 < 5:  # 左下斜线
                    overlay[y, x] = color
    return overlay

# 加载图像
gt_image = cv2.imread('data/lung_511_for_debug/images/00000011.png', cv2.IMREAD_GRAYSCALE)  # Ground Truth 图像
render_image = cv2.imread('exp/neus-blender/use_3_images_1and11and51_only_render/save/it20000-test/00000011.png', cv2.IMREAD_GRAYSCALE)  # 渲染图像

# 应用阈值化
threshold = 50
_, gt_thresh = cv2.threshold(gt_image, threshold, 255, cv2.THRESH_BINARY)  # 你可以调整阈值
_, render_thresh = cv2.threshold(render_image, threshold, 255, cv2.THRESH_BINARY)

# 提取边缘，边缘为白色，背景为黑色
gt_edges = cv2.Canny(gt_thresh, 100, 200)
# gt_edges = cv2.bitwise_not(gt_edges)  # 反转边缘图像
render_edges = cv2.Canny(render_thresh, 100, 200)
# render_edges = cv2.bitwise_not(render_edges)  # 反转边缘图像
# 将边缘图像转换为彩色图像
gt_edges_colored = cv2.cvtColor(gt_edges, cv2.COLOR_GRAY2BGR)  # 将 GT 边缘转换为彩色
render_edges_colored = cv2.cvtColor(render_edges, cv2.COLOR_GRAY2BGR)  # 将 Render 边缘转换为彩色

# 将 GT 边缘设置为红色
gt_edges_colored[:, :, 1] = 0  # G 通道置为 0
gt_edges_colored[:, :, 2] = 0  # B 通道置为 0

# 将 Render 边缘设置为蓝色
render_edges_colored[:, :, 0] = 0  # R 通道置为 0
render_edges_colored[:, :, 1] = 0  # G 通道置为 0
# 将背景设置为白色
gt_edges_colored[gt_edges == 0] = [255, 255, 255]  # 背景设置为白色
render_edges_colored[render_edges == 0] = [255, 255, 255]  # 背景设置为白色


# 计算非空洞区域的位移（使用图像质心来衡量）
gt_moments = cv2.moments(255 - gt_thresh)
render_moments = cv2.moments(255 - render_thresh)

# 计算质心
gt_cx = int(gt_moments['m10'] / gt_moments['m00'])
gt_cy = int(gt_moments['m01'] / gt_moments['m00'])

render_cx = int(render_moments['m10'] / render_moments['m00'])
render_cy = int(render_moments['m01'] / render_moments['m00'])

# 计算位移
displacement = np.sqrt((render_cx - gt_cx)**2 + (render_cy - gt_cy)**2)

# 创建带有斜线纹理的叠加图像
overlay_shape = (gt_thresh.shape[0], gt_thresh.shape[1], 3)  # 彩色图像形状
gt_moments_overlay = create_pattern_overlay(overlay_shape, gt_thresh, (255, 0, 0), direction="right")  # 红色右下斜线
render_moments_overlay = create_pattern_overlay(overlay_shape, render_thresh, (0, 0, 255), direction="left")  # 蓝色左下斜线

# 合并两张图像
combined_moments_overlay = cv2.addWeighted(gt_moments_overlay, 0.5, render_moments_overlay, 0.5, 0)
combined_thresh_overlay = cv2.addWeighted(gt_thresh, 0.5, render_thresh, 0.5, 0)
combined_edge_overlay = cv2.addWeighted(gt_edges_colored, 0.5, render_edges_colored, 0.5, 0)


# 在图像上绘制质心和位移信息
fig, axes = plt.subplots(3, 3, figsize=(18, 18))

# 显示 GT 原图
axes[0, 0].imshow(gt_image, cmap='gray')
axes[0, 0].scatter(gt_cx, gt_cy, color='red', label='GT Centroid')
axes[0, 0].set_title('GT Original with Centroid')
axes[0, 0].legend()
axes[0, 0].axis('off')

# 显示 Render 原图
axes[0, 1].imshow(render_image, cmap='gray')
axes[0, 1].scatter(render_cx, render_cy, color='blue', label='Render Centroid')
axes[0, 1].set_title('Render Original with Centroid')
axes[0, 1].legend()
axes[0, 1].axis('off')

# 显示叠加图像
axes[0, 2].imshow(combined_moments_overlay)
axes[0, 2].set_title('Overlay (GT: Red, Render: Blue)')
axes[0, 2].text(10, 30, f'Displacement: {displacement:.2f} px', color='yellow', fontsize=12, bbox=dict(facecolor='black', alpha=0.5))
axes[0, 2].axis('off')

# 显示 GT 阈值化结果
axes[1, 0].imshow(gt_thresh, cmap='gray')
axes[1, 0].set_title('GT Thresholded')
axes[1, 0].axis('off')

# 显示 Render 阈值化结果
axes[1, 1].imshow(render_thresh, cmap='gray')
axes[1, 1].set_title('Render Thresholded')
axes[1, 1].axis('off')

# 显示空白占位图
axes[1, 2].imshow(combined_thresh_overlay, cmap='gray')
axes[1, 2].set_title('Overlay ')
axes[1, 2].axis('off')


# 显示 GT 边缘
axes[2, 0].imshow(cv2.cvtColor(gt_edges_colored, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供 Matplotlib 显示
axes[2, 0].set_title('GT Edges (Red)')
axes[2, 0].axis('off')

# 显示 Render 边缘
axes[2, 1].imshow(cv2.cvtColor(render_edges_colored, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供 Matplotlib 显示
axes[2, 1].set_title('Render Edges (Blue)')
axes[2, 1].axis('off')

# 显示叠加边缘图像
axes[2, 2].imshow(cv2.cvtColor(combined_edge_overlay, cv2.COLOR_BGR2RGB))  # 转换为 RGB 格式以供 Matplotlib 显示
axes[2, 2].set_title('Overlay Edges (Red + Blue)')
axes[2, 2].axis('off')

plt.tight_layout()
plt.show()