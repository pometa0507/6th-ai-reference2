import sys

import numpy as np
import torch
from google.protobuf import text_format

sys.path.append('../second.pytorch/')
from second.protos import pipeline_pb2
from second.pytorch.train import build_network
from second.utils import config_tool


class Second3DDector(object):

    def __init__(self, config_p, model_p, device="cpu"):
        
        self.config_p = config_p
        self.model_p = model_p
        self.device = device
        self._init_model()

    def _init_model(self):

        self.config = pipeline_pb2.TrainEvalPipelineConfig()
        with open(self.config_p, 'r') as f:
            proto_str = f.read()
            text_format.Merge(proto_str, self.config)
        
        self.input_cfg = self.config.eval_input_reader
        self.max_voxels = self.input_cfg.preprocess.max_number_of_voxels
        self.model_cfg = self.config.model.second
        self.num_points_feature = self.model_cfg.num_point_features

        self.net = build_network(self.model_cfg).to(self.device).eval()
        self.net.load_state_dict(torch.load(self.model_p))
        
        self.target_assigner = self.net.target_assigner
        self.voxel_generator = self.net.voxel_generator

        grid_size = self.voxel_generator.grid_size
        feature_map_size = grid_size[:2] // \
                           config_tool.get_downsample_factor(self.model_cfg)
        feature_map_size = [*feature_map_size, 1][::-1]

        self.anchors = self.target_assigner.generate_anchors(feature_map_size)['anchors']
        self.anchors = torch.tensor(self.anchors, dtype=torch.float32, device=self.device)
        self.anchors = self.anchors.view(1, -1, 7)

    def load_pc(self, pc_f):

        if type(pc_f) is str:
            points = np.fromfile(pc_f, dtype=np.float32, count=-1)
            points = points.reshape([-1, self.num_points_feature])
        elif type(pc_f) is np.ndarray:
            points = pc_f
        return points

    def load_an_in_example_from_points(self, points):

        res = self.voxel_generator.generate(points, max_voxels=self.max_voxels)
        voxels = res['voxels']
        coords = res['coordinates']
        num_points = res['num_points_per_voxel']
        
        coords = np.pad(coords, ((0,0), (1,0)), mode='constant', constant_values=0)
        voxels = torch.tensor(voxels, dtype=torch.float32, device=self.device)
        coords = torch.tensor(coords, dtype=torch.int32, device=self.device)
        num_points = torch.tensor(num_points, dtype=torch.int32, device=self.device)
        return {
            'anchors': self.anchors,
            'voxels': voxels,
            'num_points': num_points,
            'coordinates': coords,
        }

    def predict_on_points(self, points):

        points = self.load_pc(points)

        example = self.load_an_in_example_from_points(points)
        pred = self.net(example)[0]
        boxes_lidar = pred['box3d_lidar'].detach().cpu().numpy()
        scores = pred["scores"].detach().cpu().numpy()
        label_preds = pred["label_preds"].detach().cpu().numpy()

        return {
            'boxes_lidar': boxes_lidar,
            'scores': scores,
            'label_preds': label_preds,
        }
