from PIL import Image
from mpl_toolkits.mplot3d import Axes3D, art3d
import matplotlib.pyplot as plt
import numpy as np
import json
import os

def plot_trajectory_with_images(json_path, image_root, downsample=16, scale=0.001):
    with open(json_path, 'r') as f:
        data = json.load(f)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    for i, frame in enumerate(data["frames"]):
        transform = np.array(frame["transform_matrix"])
        pos = transform[:3, 3]
        R = transform[:3, :3]
        z_axis = R[:, 2] / np.linalg.norm(R[:, 2]) * 0.25 # 相机光轴方向

        # 绘制相机位置
        ax.scatter(*pos, c='r')
        ax.quiver(pos[0], pos[1], pos[2],
                z_axis[0], z_axis[1], z_axis[2],
                length=scale * 3, color='b')

        # 加载图像
        img_path = os.path.join(image_root, frame["file_path"] +  ".png")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found")
            continue

        img = Image.open(img_path)
        img = img.resize((img.width // downsample, img.height // downsample))
        img = np.asarray(img) / 255.0  # 归一化
        if img.ndim == 2:  # 灰度图转为RGB
            img = np.stack([img]*3, axis=-1)
        elif img.shape[2] == 4:  # 去掉 alpha 通道
            img = img[:, :, :3]

        # 准备图像平面（xy平面上一个小方块）
        h, w, _ = img.shape
        x = np.linspace(-0.5, 0.5, w) * scale
        y = np.linspace(-0.5, 0.5, h) * scale
        x, y = np.meshgrid(x, y)
        z = np.zeros_like(x)

        # 将图像平面旋转到相机面朝方向
        xyz = np.stack([x, y, z], axis=-1).reshape(-1, 3)  # (H*W, 3)
        rotated = xyz @ R.T + pos  # (H*W, 3)

        X = rotated[:, 0].reshape(h, w)
        Y = rotated[:, 1].reshape(h, w)
        Z = rotated[:, 2].reshape(h, w)

        ax.plot_surface(X, Y, Z, rstride=1, cstride=1, facecolors=img, shade=False)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title("Camera Trajectory with Image Thumbnails")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    json_path = "data/lung_511_for_debug/transforms_val.json"
    image_root = "data/lung_511"  # 根据你的实际路径修改

    plot_trajectory_with_images(json_path, image_root)
