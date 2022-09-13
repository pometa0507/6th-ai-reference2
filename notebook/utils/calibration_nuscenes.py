import copy

import numpy as np
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import view_points
from pyquaternion import Quaternion


def lidar_to_global(points,
                    lidar_ego_pose,
                    lidar_calibration):
    """
    Lidar座標 -> Global座標 に変換

    Args:
        points: Lidar座標の点群 shape = (N, C)
        lidar_ego_pose: Lidarのegoポーズパラメータ
        lidar_calibration: Lidarのキャリブレーションパラメータ
    Returns:
        points_global: Glabal座標の点群 shape = (N, C)
    """

    points_tmp = copy.deepcopy(points)
    num_point_feature = points_tmp.shape[1]
    assert num_point_feature >= 3, f"num_point_feature: {num_point_feature} は3以上必要です。"
    
    if num_point_feature == 3:  # (N, 3) -> (N, 4)
        points_tmp = np.hstack([points_tmp, np.zeros([points_tmp.shape[0], 1])])
        
    elif num_point_feature >= 4:
        points_feature = copy.deepcopy(points[:, 3:])
        points_tmp = points_tmp[:, :4]
    
    pc = LidarPointCloud(points_tmp.T)
    
    # Lidar座標系 -> カメラ座標系 に変換
    # 1st step: Lidar座標 -> Lidar車両ポーズ
    pc.rotate(Quaternion(lidar_calibration['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibration['translation']))
    
    # 2nd step: Lidar車両ポーズ -> Global座標
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))

    points_global = pc.points[:3, :].T        # (N, 3)  N:(x, y, z)
    if num_point_feature >= 4:
        points_global = np.hstack([points_global, points_feature])
    
    return points_global
    
    
def lidar_to_cam(points,
                 cam_ego_pose,
                 lidar_ego_pose,
                 cam_calibration,
                 lidar_calibration):
    """
    Lidar座標 -> カメラ座標 に変換

    Args:
        points: Lidar座標の点群 shape = (N, C)
        cam_ego_pose: カメラのegoポーズパラメータ
        lidar_ego_pose: Lidarのegoポーズパラメータ
        cam_calibration: カメラのキャリブレーションパラメータ
        lidar_calibration: Lidarのキャリブレーションパラメータ
    Returns:
        points_cam: カメラ座標の点群 shape = (N, C)
    """

    points_tmp = copy.deepcopy(points)
    num_point_feature = points_tmp.shape[1]
    assert num_point_feature >= 3, f"num_point_feature: {num_point_feature} は3以上必要です。"
    
    if num_point_feature == 3:  # (N, 3) -> (N, 4)
        points_tmp = np.hstack([points_tmp, np.zeros([points_tmp.shape[0], 1])])
        
    elif num_point_feature >= 4:
        points_feature = copy.deepcopy(points_tmp[:, 3:])
        points_tmp = points_tmp[:, :4]
    
    pc = LidarPointCloud(points_tmp.T)
    
    # Lidar座標系 -> カメラ座標系 に変換
    # 1st step: Lidar座標 -> Lidar車両ポーズ
    pc.rotate(Quaternion(lidar_calibration['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_calibration['translation']))
    
    # 2nd step: Lidar車両ポーズ -> Global座標
    pc.rotate(Quaternion(lidar_ego_pose['rotation']).rotation_matrix)
    pc.translate(np.array(lidar_ego_pose['translation']))
    
    # 3rd step: Global座標 -> カメラ車両ポーズ
    pc.translate(-np.array(cam_ego_pose['translation']))
    pc.rotate(Quaternion(cam_ego_pose['rotation']).rotation_matrix.T)
    
    # 4th step: カメラ車両ポーズ -> カメラ座標系
    pc.translate(-np.array(cam_calibration['translation']))
    pc.rotate(Quaternion(cam_calibration['rotation']).rotation_matrix.T)

    points_cam = pc.points[:3, :].T        # (N, 3)  N:(x, y, z)
    
    if num_point_feature >= 4:
        points_cam = np.hstack([points_cam, points_feature])
    
    return points_cam

def get_image_points(points, view):
    """
    点群を画像へ投影し画像座標を取得

    Args:
        points: Lidar座標の点群 shape = (N, 3) [x, y, z]
        view: プロジェクション行列
    Returns:
        points_projected: 画像座標の点群 shape = (N, 3) [x, y, z]
    """
    assert points.shape[1] == 3
    
    # Projection matrix from (3, 3) to (4, 4)
    viewpad = np.eye(4)
    viewpad[:view.shape[0], :view.shape[1]] = view
    
    nbr_points = points.shape[0]
    
    # Do operation in homogeneous coordinates
    points_hm = np.hstack((points, np.ones((nbr_points, 1))))
    # image points = Projection @ Points
    points_projected = viewpad @ points_hm.T    # (4, N) = (4, 4) @ (4, N)
    points_projected = points_projected / points_projected[2, :]
    
    # get x, y, z (x and y are coordinates on the image, and z = 1)
    points_projected = points_projected[:3, ].T       # (N, 3)  N:(x, y, z=1)
    points_projected = np.floor(points_projected).astype(np.int32) # 小数点以下切り捨て
    
    return points_projected