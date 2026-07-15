
import json
import math

import numpy as np
from scipy.spatial.transform import Rotation

# from USD camera convention to World camera convention
W_U_TRANSFORM = np.array([[0, 0, -1, 0], [-1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
# from World camera convention to USD camera convention
U_W_TRANSFORM = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])



def convert_camera_intrinsics_colmap(camera_intrinsics, width=1280, height=720):
    # 提取相机内参的元素
    fx = camera_intrinsics[0, 0]
    fy = camera_intrinsics[1, 1]
    cx = camera_intrinsics[0, 2]
    cy = camera_intrinsics[1, 2]
    # 格式化字符串，确保使用双引号
    camera_info = f'1 PINHOLE {width} {height} {fx} {fy} {cx} {cy}'
    return camera_info

def camera_intrinsics_calc(focal_length, horizontal_aperture, vertical_aperture, width=1280, height=720):
    # 相机内参 疑问：和isaacsim获取的内参不一样 isaac的内参f_x=f_y 待定...
    # focal_length = 0.1809999942779541  # 焦距，单位：米
    # horizontal_aperture = 0.38399999141693115  # 水平光圈，单位：米
    # vertical_aperture = 0.24000000953674316  # 垂直光圈，单位：米
    resolution_width = width  # 图像宽度
    resolution_height = height  # 图像高度

    # 假设传感器尺寸（使用光圈大小一般设置）来计算像素尺寸
    # 这里假设每像素在焦距上的计算需要应用这些参数
    pixel_width = horizontal_aperture / resolution_width  # 每个像素的物理宽度
    pixel_height = vertical_aperture / resolution_height  # 每个像素的物理高度

    # 焦距转换为像素单位
    f_x = focal_length / pixel_width  # 水平焦距（像素）
    f_y = focal_length / pixel_height  # 垂直焦距（像素）

    # 主点坐标
    c_x = resolution_width / 2
    c_y = resolution_height / 2

    # 构造内参矩阵
    K = np.array([[f_x, 0, c_x],
                  [0, f_y, c_y],
                  [0, 0, 1]])
    return K

def rotation_matrix_to_quaternion(R):
    qw = np.sqrt(1.0 + R[0, 0] + R[1, 1] + R[2, 2]) / 2.0
    qx = (R[2, 1] - R[1, 2]) / (4.0 * qw)
    qy = (R[0, 2] - R[2, 0]) / (4.0 * qw)
    qz = (R[1, 0] - R[0, 1]) / (4.0 * qw)

    if qw == 0.0:
        qx = 1.0
        qy = 0.0
        qz = 0.0

    return np.array([qw, qx, qy, qz])


def load_trajectory(file_path):
    """
    读取轨迹 JSON 文件并返回轨迹点列表。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data['trajectory']

def load_trajectory_traj_type(file_path, traj_type):
    """
    读取轨迹 JSON 文件并返回轨迹点列表。
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[traj_type]['trajectory']


def compute_yaw(p1, p2):
    """
    计算从 p1 到 p2 的 yaw 角度。
    """
    dx = p2['x'] - p1['x']
    dy = p2['y'] - p1['y']
    yaw = math.atan2(dy, dx)  # 返回值范围 [-pi, pi]
    return yaw


def rotation_matrix_from_yaw(yaw):
    """
    根据 yaw 角度生成旋转矩阵。
    仅考虑绕 Z 轴的旋转，保持相机水平。
    """
    return np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])


def quaternion_from_yaw(yaw):
    """
    根据 yaw 角度生成四元数。
    """
    print(yaw)
    print(type(yaw))
    rotation = Rotation.from_euler('z', yaw)
    return rotation.as_quat()  # [x, y, z, w]


def compute_camera_poses(trajectory):
    """
    对于每个轨迹点，计算相机位姿 R, T, 四元数。
    返回一个列表，每个元素包含 R, T, quaternion。
    """
    poses = []
    yaws = []
    num_points = len(trajectory)

    for i in range(num_points):
        current_point = trajectory[i]
        T = np.array([current_point['x'], current_point['y'], -0.65])

        if i < num_points - 1:
            next_point = trajectory[i + 1]
            yaw = compute_yaw(current_point, next_point)
        elif i > 0:
            # 对于最后一个点，使用前一个点的 yaw
            prev_point = trajectory[i - 1]
            yaw = compute_yaw(prev_point, current_point)
        else:
            # 只有一个点，默认 yaw 为 0
            yaw = 0.0

        R_matrix = rotation_matrix_from_yaw(yaw)
        quaternion = quaternion_from_yaw(yaw)  # [x, y, z, w]

        pose = {
            'R': R_matrix.tolist(),
            'T': T.tolist(),
            'quaternion': quaternion.tolist()
        }
        poses.append(pose)
        yaws.append(yaw)

    return poses, yaws


def radians_to_degrees(radians):
    degrees = radians * (180 / math.pi)  # 或者可以使用 math.degrees(radians)
    return degrees



def rot_matrices_to_quats(rotation_matrices: np.ndarray, device=None) -> np.ndarray:
    """Vectorized version of converting rotation matrices to quaternions

    Args:
        rotation_matrices (np.ndarray): N Rotation matrices with shape (N, 3, 3) or (3, 3)

    Returns:
        np.ndarray: quaternion representation of the rotation matrices (N, 4) or (4,) - scalar first
    """
    rot = Rotation.from_matrix(rotation_matrices)
    result = rot.as_quat()
    if len(result.shape) == 1:
        result = result[[3, 0, 1, 2]]
    else:
        result = result[:, [3, 0, 1, 2]]
    return result

def quats_to_rot_matrices(quaternions: np.ndarray, device=None) -> np.ndarray:
    """Vectorized version of converting quaternions to rotation matrices

    Args:
        quaternions (np.ndarray): quaternions with shape (N, 4) or (4,) and scalar first

    Returns:
        np.ndarray: N Rotation matrices with shape (N, 3, 3) or (3, 3)
    """
    if len(quaternions.shape) == 1:
        q = quaternions[[1, 2, 3, 0]]
    else:
        q = quaternions[:, [1, 2, 3, 0]]
    rot = Rotation.from_quat(q)
    result = rot.as_matrix()
    return result

def create_tensor_from_list(data, dtype, device=None):
    return as_type(np.array(data), dtype)

def as_type(data, dtype):
    if dtype == "float32":
        return data.astype(np.float32)
    elif dtype == "bool":
        return data.astype(bool)
    elif dtype == "int32":
        return data.astype(np.int32)
    elif dtype == "int64":
        return data.astype(np.int64)
    elif dtype == "long":
        return data.astype(np.long)
    elif dtype == "uint8":
        return data.astype(np.uint8)
    else:
        print(f"Type {dtype} not supported.")

def convert(data, device=None, dtype="float32", indexed=None):
    return as_type(np.asarray(data), dtype)

def world_to_camera_orientation(camera_world_orientation_w):
    camera_world_orientation_w = convert(camera_world_orientation_w, device=None)
    world_w_cam_w_R = quats_to_rot_matrices(camera_world_orientation_w)
    w_u_R = create_tensor_from_list(
        W_U_TRANSFORM[:3, :3].tolist(), dtype="float32"
    )
    calc_w_to_usd_orientation = rot_matrices_to_quats(np.matmul(world_w_cam_w_R, w_u_R))
    # print(f'world.calc_w_to_usd_orientation = {calc_w_to_usd_orientation}')
    return calc_w_to_usd_orientation

def camera_to_world_orientation(camera_world_orientation_usd):
    world_w_cam_u_R = quats_to_rot_matrices(camera_world_orientation_usd)
    u_w_R = create_tensor_from_list(
        U_W_TRANSFORM[:3, :3].tolist(), dtype="float32"
    )
    calc_orientation = rot_matrices_to_quats(np.matmul(world_w_cam_u_R, u_w_R))
    # print(f'world.calc_orientation = {calc_orientation}')
    return calc_orientation

def quaternion_to_rotation_matrix(q):
    """从四元数转换为旋转矩阵"""
    q = q / np.linalg.norm(q)  # 归一化
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y**2 + z**2), 2 * (x * y - w * z), 2 * (x * z + w * y), 0],
        [2 * (x * y + w * z), 1 - 2 * (x**2 + z**2), 2 * (y * z - w * x), 0],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x**2 + y**2), 0],
        [0, 0, 0, 1]
    ])

# 4. 创建平移矩阵
def translation_matrix(translation):
    """创建平移矩阵"""
    mat = np.eye(4)  # 创建 4x4 单位矩阵
    mat[:3, 3] = translation  # 设置平移
    return mat

# 7. 正交化矩阵（即确保每列都是单位向量且相互垂直）
def orthonormalize(mat):
    """正交化矩阵"""
    q, r = np.linalg.qr(mat[:3, :3])  # QR 分解
    return q

def rotation_matrix_to_euler(mat):
    """从旋转矩阵提取欧拉角（以度为单位）"""
    sy = np.sqrt(mat[0, 0] * mat[0, 0] +  mat[1, 0] * mat[1, 0])
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(mat[2, 1] , mat[2, 2])
        y = np.arctan2(-mat[2, 0], sy)
        z = np.arctan2(mat[1, 0], mat[0, 0])
    else:
        x = np.arctan2(-mat[1, 2], mat[1, 1])
        y = np.arctan2(-mat[2, 0], sy)
        z = 0
    return np.array([np.degrees(x), np.degrees(y), np.degrees(z)])

def is_yaw_value_valid(yaw_value):
    # 定义需要检查的有效值
    valid_values = [0.0, 90.0, -90.0, 180.0, -180.0, 270.0, -270.0, 360.0]

    # 检查 yaw_value 是否与 valid_values 中的任何一个值接近
    for valid in valid_values:
        if math.isclose(yaw_value, valid, abs_tol=1e-3):  # 自定义公差，这里设置为1e-3
            return True

    return False

def xyz_euler_trans_gs_colmap(x, y, z, roll, pitch, yaw, degree_flag=False):

    if degree_flag:
        pitch_value = pitch
        roll_value = roll
        yaw_value = yaw
    else:
        pitch_value = radians_to_degrees(pitch)
        roll_value = radians_to_degrees(roll)
        yaw_value = radians_to_degrees(yaw)

    # 设置相机local的位置 (x, y, z)
    gs_T = np.array([x, y, z])
    if is_yaw_value_valid(yaw_value):
        euler_angles_w = np.array([roll_value, pitch_value, yaw_value + 0.001])
    else:
        euler_angles_w = np.array([roll_value, pitch_value, yaw_value])
    RR = Rotation.from_euler('xyz', euler_angles_w, degrees=True).as_matrix()
    # print(f'RR = {RR}')
    quaternion_w = rotation_matrix_to_quaternion(RR)
    # print(f'quaternion_w = {quaternion_w}')

    quaternion_usd = world_to_camera_orientation(quaternion_w)

    # 将 numpy 数组转换为 Gf.Vec3d 对象
    vec3d_pose = np.array([gs_T[0], gs_T[1], gs_T[2]])
    # 将 numpy 数组转换为 Gf.Quatf 对象（假设你的四元数是浮点数）
    quat = np.array([float(quaternion_usd[0]), float(quaternion_usd[1]), float(quaternion_usd[2]),
                     float(quaternion_usd[3])])

    rotation_matrix = quaternion_to_rotation_matrix(quat).T
    translation_mat = translation_matrix(vec3d_pose).T

    # 5. 构建 Camera to World 变换矩阵
    camera_to_world_mat = rotation_matrix @ translation_mat

    # 6. 提取相机到物体的矩阵和位置
    camera_to_object_mat = camera_to_world_mat

    camera_to_object_pos = camera_to_object_mat[-1, :3]  # 提取位置

    # camera_to_object_mat_qua = rotation_matrix_to_quaternion(camera_to_object_mat[:3, :3])

    camera_to_object_mat_rot = camera_to_object_mat[:3, :3].T

    # 8. 提取旋转部分
    camera_to_object_rot = rotation_matrix_to_euler(camera_to_object_mat_rot)

    pose_data = {
        'position': list(camera_to_object_pos),
        'rotation': list(np.deg2rad(camera_to_object_rot))
    }

    position_t = np.array(pose_data['position']),
    euler_angles = np.array(pose_data['rotation'])

    C2W = np.eye(4)
    C2W[:3, :3] = Rotation.from_euler('xyz', euler_angles).as_matrix()
    C2W[:3, 3] = position_t[0]
    W2C = np.linalg.inv(C2W)
    ISAAC_SIM_TO_GS_CONVENTION = np.array([
        [1, 0, 0, 0],
        [0, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 0, 1]
    ])

    W2C = ISAAC_SIM_TO_GS_CONVENTION @ W2C

    R = W2C[:3, :3].T.T
    T = W2C[:3, 3]

    quaternion = rotation_matrix_to_quaternion(R)

    camera_id = 1

    image_id = 1
    image_name = f"image_{1}.jpg"

    camera_image_pose_colmap = f'{image_id} ' + " ".join(map(str, quaternion)) + " " + " ".join(
        map(str, T)) + " " + f'{camera_id} {image_name}'

    return camera_image_pose_colmap

def main():
    pitch = 0.0
    roll = 0.0
    yaw = math.pi / 2
    x = 0.0
    y = 0.0
    z = 0.0
    camera_image_pose_colmap = xyz_euler_trans_gs_colmap(x, y, z, roll, pitch, yaw, False)
    print(f"camera_image_pose_colmap = {camera_image_pose_colmap}")


if __name__ == '__main__':
    main()