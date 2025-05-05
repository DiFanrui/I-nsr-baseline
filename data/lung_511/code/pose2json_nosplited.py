import numpy as np
import json
from pathlib import Path

def quaternion_to_rotation_matrix(q):
    qx, qy, qz, qw = q
    R = np.array([
        [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
        [2*(qx*qy + qz*qw),     1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
        [2*(qx*qz - qy*qw),         2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
    ])
    return R

def make_transform_matrix(R, t):
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T

def read_poses(file_path, sep=' ', matrix_shape=(8, ), trajectory_indices=(slice(None), slice(1, 4)), orientation_indices=(slice(None), slice(4, 8))):
    """
    通用函数读取位姿数据并提取轨迹点。
    
    参数:
        file_path (str): 文件路径。
        sep (str): 数据分隔符，默认为 ','。
        matrix_shape (tuple): 数据的形状，例如 (4, 4) 表示 4x4 矩阵。
        trajectory_indices (tuple): 提取轨迹点的索引，例如 (slice(None), slice(None), 3) 表示提取 4x4 矩阵的最后一列。
    
    返回:
        trajectory (np.ndarray): 提取的轨迹点。
    """
    poses = []
    with open(file_path, 'r') as f:
        for line in f:
            # 跳过以 '#' 开头的注释行
            if line.startswith('#'):
                continue
            if line.strip():
                pose = np.fromstring(line, sep=sep).reshape(matrix_shape)
                poses.append(pose)
    
    poses = np.array(poses)  # 转换为数组
    trajectory = poses[trajectory_indices]  # 提取轨迹点
    orientations = None
    if orientation_indices is not None:
        orientations = poses[orientation_indices]  # 提取相机朝向
    return trajectory, orientations


def convert_to_blender_format(pose_file_path, image_dir, image_extension=".png", image_width=480):
    # 你的相机内参
    fx = 447.679079249249
    cx = 252.560061902772
    camera_angle_x = 2 * np.arctan(image_width / (2 * fx))

    frames = []

    # 读取位姿文件
    trajectory, orientations = read_poses(pose_file_path)

    for idx, (t, q) in enumerate(zip(trajectory, orientations)):
        R = quaternion_to_rotation_matrix(q)
        T = make_transform_matrix(R, t)

        # OpenCV to Blender 坐标系转换：y 和 z 轴翻转
        T[:3, 1:3] *= -1

        frame = {
            "file_path": f"{image_dir}/{idx:08d}",  # 你图像名如果是0000.jpg形式
            "transform_matrix": T.tolist()
        }
        frames.append(frame)

    output = {
        "camera_angle_x": camera_angle_x,
        "frames": frames
    }



    with open("/home/data1/difanrui/Project/instant-nsr-pl/data/blender_type/transforms.json", "w") as f:
        json.dump(output, f, indent=4)
    
    print("✅ transforms.json 生成完毕！")


# 示例调用（你需要修改路径）
convert_to_blender_format(
    pose_file_path="/home/data1/difanrui/Project/instant-nsr-pl/data/blender_type/groundtruth.txt",     # 位姿文件
    image_dir="/home/data1/difanrui/Project/instant-nsr-pl/data/blender_type/images",           # 相对路径：相对于 transforms.json 的图像路径
    image_extension=".png",         # 图像格式
    image_width=480                # 图像宽度（用于计算水平视角）
)
