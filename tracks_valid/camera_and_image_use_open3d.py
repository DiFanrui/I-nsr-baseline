import open3d as o3d
import numpy as np
import json
import os
from PIL import Image

def plot_trajectory_with_images_open3d(json_path, image_root, downsample=2, scale=0.001):
    with open(json_path, 'r') as f:
        data = json.load(f)

    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # 创建一个点云对象用于存储相机位置
    camera_positions = o3d.geometry.PointCloud()

    for i, frame in enumerate(data["frames"]):
        transform = np.array(frame["transform_matrix"])
        pos = transform[:3, 3]
        R = transform[:3, :3]
        z_axis = R[:, 2] / np.linalg.norm(R[:, 2]) * 0.0005  # 相机光轴方向

        # 添加相机位置到点云
        camera_positions.points.append(pos)

        # 添加光轴方向（z_axis）作为线段
        line_set = o3d.geometry.LineSet()
        line_set.points = o3d.utility.Vector3dVector([pos, pos + z_axis])  # 起点为相机位置，终点为光轴方向
        line_set.lines = o3d.utility.Vector2iVector([[0, 1]])  # 定义线段
        line_set.colors = o3d.utility.Vector3dVector( [[0, 0, 1]])  # 蓝色线段
        vis.add_geometry(line_set)

        # 加载图像
        img_path = os.path.join(image_root, frame["file_path"] + ".png")
        if not os.path.exists(img_path):
            print(f"Warning: {img_path} not found")
            continue

        img = Image.open(img_path)
        img = img.resize((img.width // downsample, img.height // downsample))
        img = np.asarray(img) / 255.0  # 归一化
        if img.ndim == 2:  # 灰度图转为RGB
            img = np.stack([img] * 3, axis=-1)
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

        # 创建一个 open3d 的三角网格用于显示图像
        vertices = rotated.reshape(-1, 3)
        triangles = []
        for row in range(h - 1):
            for col in range(w - 1):
                idx = row * w + col
                triangles.append([idx, idx + 1, idx + w])
                triangles.append([idx + 1, idx + w + 1, idx + w])

        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(triangles)
        mesh.vertex_colors = o3d.utility.Vector3dVector(img.reshape(-1, 3))

        # 设置双面显示
        mesh.triangle_material_ids = o3d.utility.IntVector([0] * len(triangles))
        vis.get_render_option().mesh_show_back_face = True  # 允许双面显示

        vis.add_geometry(mesh)

    # 添加相机位置点云
    camera_positions.paint_uniform_color([1, 0, 0])  # 红色
    vis.add_geometry(camera_positions)

    vis.run()
    vis.destroy_window()


if __name__ == "__main__":
    json_path = "data/lung_511_for_debug/transforms_val.json"
    image_root = "data/lung_images_with_id"  # 根据你的实际路径修改

    plot_trajectory_with_images_open3d(json_path, image_root)