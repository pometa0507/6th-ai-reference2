import numpy as np
import torch
from nuscenes.utils.data_classes import LidarPointCloud
from PIL import Image
from torchvision import transforms

from .calibration_nuscenes import (get_image_points, lidar_to_cam,
                                   lidar_to_global)


######################
# Based on https://github.com/Song-Jingyu/PointPainting
# Licensed under The MIT License
# Authors Chen Gao, Jingyu Song, Youngsun Wi, Zeyu Wang
######################
def get_segmentation_score(img_path, model, device):
    """
    画像に対するセグメンテーションのスコアを出力する

    Returns
    -------
    output_reassign_softmax : modelの出力(torch.tensor)
        a tensor H  * W * 3, for each pixel we have 3 scorer that sums to 1
    """

    input_image = Image.open(img_path)
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0)  # バッチ軸追加

    input_batch = input_batch.to(device)

    with torch.no_grad():
        output = model(input_batch)['out'][0]

    output_permute = output.permute(1, 2, 0)
    output_probability, output_predictions = output_permute.max(2)

    other_object_mask = ~((output_predictions == 0) | (output_predictions == 7) | \
                          (output_predictions == 15))
    detect_object_mask = ~other_object_mask
    sf = torch.nn.Softmax(dim=2)

    # car = 7, person = 15, background = 0
    output_reassign = torch.zeros(
        output_permute.size(0), output_permute.size(1), 3)
    output_reassign[:, :, 0] = detect_object_mask * output_permute[:, :, 0] + \
                               other_object_mask * output_probability  # background
    output_reassign[:, :, 1] = output_permute[:, :, 7]  # car
    output_reassign[:, :, 2] = output_permute[:, :, 15] # person
    output_reassign_softmax = sf(output_reassign).cpu().numpy()

    return output_reassign_softmax


######################
# Based on https://github.com/nutonomy/nuscenes-devkit
#  https://github.com/nutonomy/nuscenes-devkit/blob/master/python-sdk/nuscenes/nuscenes.py#L834
# Licensed under The Apache-2.0 License
######################
def map_pointcloud_to_image(cam_path,
                            lidar_path,
                            cam_ego_pose,
                            lidar_ego_pose,
                            cam_calibration,
                            lidar_calibration,
                            min_dist=1.0):
    """
    Lidar点群を画像に投影して画像座標を取得

    Returns:
        points_mask: 画像範囲でマスクした点群 shape=(M, 4) [x, y, z, intensity]
        points_image: 画像座標の点群  shape=(M, 2) [x, y]
    """

    # 画像の読み込み
    img = Image.open(cam_path)

    # 点群データの読み込み
    pc = LidarPointCloud.from_file(lidar_path) 
    points = pc.points.T    # (N, 4)

    # Lidar座標系 -> カメラ座標系 に変換
    points_cam = lidar_to_cam(points,
                            cam_ego_pose,
                            lidar_ego_pose,
                            cam_calibration,
                            lidar_calibration)

    # 点群を画像へ投影し画像上の座標を取得
    projection = np.array(cam_calibration['camera_intrinsic'])
    points_image = get_image_points(points_cam[:, :3], projection) # (N, 3)

    depths = points_cam[:, 2]  # (N, )
    min_dist = 1.0

    # マスク処理
    mask = np.ones(depths.shape[0], dtype=bool)  
    # depth(深さ)がmin_dist以上
    mask = np.logical_and(mask, depths > min_dist)
    # xの値が0～W(画像の幅)
    mask = np.logical_and(mask, points_image[:, 0] > 0)
    mask = np.logical_and(mask, points_image[:, 0] < img.size[0])
    # yの値が0～H(画像の高さ)
    mask = np.logical_and(mask, points_image[:, 1] > 0)
    mask = np.logical_and(mask, points_image[:, 1] < img.size[1])

    # マスク後の点群（Lidar座標）
    points_mask = points[mask, :]  # (M, 4)  4: x, y, z, intensity

    # マスク後の点群（画像上の座標）
    points_image = points_image[mask, :2]  # (M, 2)   2: x, y    
     
    return points_mask, points_image


def overlap_seg(img,
                class_scores,
                palette=None,
                opacity=0.5):
    """
    Draw `class_scores` over `img`
    """

    if palette is None:
        palette = [[255, 255, 255], # background
                   [253, 141, 60],  # car
                   [0, 0, 255],     # person
                  ]

    result = np.argmax(class_scores, 2) 
    seg = result
    
    if palette is None:
        palette = np.random.randint(
            0, 255, size=(class_scores.shape[2], 3))

    palette = np.array(palette)
    assert 0 < opacity <= 1.0

    color_seg = np.zeros((seg.shape[0], seg.shape[1], 3), dtype=np.uint8)
    
    for label, color in enumerate(palette):      
        color_seg[seg == label, :] = color
    
    img = img * (1 - opacity) + color_seg * opacity
    img = img.astype(np.uint8)

    return img
    