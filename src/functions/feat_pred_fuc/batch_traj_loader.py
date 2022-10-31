import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
import pickle
from pathlib import Path
import itertools
import random
import pdb
from tqdm import tqdm
import msgpack_numpy
import torch
from src.utils.sim_utils import set_up_habitat
from torch_geometric.data import Data


class Loader:
    def __init__(self, args):
        self.args = args
        np.random.seed(args.seed)
        self.datasets = {}
        self.feat_size = 512
        self.edge_feat_size = 16

    def geo_dist(self, ind1, goalind, data_obj):
        geodesic = 0
        if ind1 <= goalind:
            ranges = range(ind1, goalind)
        else:
            ranges = range(goalind, ind1)
        for i in ranges:
            geodesic += np.linalg.norm(
                np.asarray(data_obj["nodes"]["pos"][i])
                - np.asarray(data_obj["nodes"]["pos"][i + 1])
            )
        return geodesic

    def geo_dist_valid(self, index, goalind, node_pos, data_obj):
        geodesic = 0
        if index == goalind:
            geodesic += np.linalg.norm(
                np.asarray(node_pos[-1]) - np.asarray(data_obj["nodes"]["pos"][goalind])
            )
        elif index < goalind:
            geodesic += np.linalg.norm(
                np.asarray(node_pos[-1])
                - np.asarray(data_obj["nodes"]["pos"][index + 1])
            )
            for i in range(index + 1, goalind):
                geodesic += np.linalg.norm(
                    np.asarray(data_obj["nodes"]["pos"][i])
                    - np.asarray(data_obj["nodes"]["pos"][i + 1])
                )
        else:
            geodesic += np.linalg.norm(
                np.asarray(node_pos[-1])
                - np.asarray(data_obj["nodes"]["pos"][index - 1])
            )
            for i in range(goalind, index - 1):
                geodesic += np.linalg.norm(
                    np.asarray(data_obj["nodes"]["pos"][i])
                    - np.asarray(data_obj["nodes"]["pos"][i + 1])
                )
        return geodesic
    def true_node_to_point(self, index, valid_points, goal_index, data_obj):
        """add graph info and valid node info"""
        node_ids = np.asarray(data_obj["nodes"]["ids"][0:index])
        node_pos = np.asarray(data_obj["nodes"]["pos"][0:index])
        node_feat = np.asarray(data_obj["nodes"]["feat"][0:index])

        adj_matrix = []
        edge_feats = []
        edge_pos = []
        for i in range(0, len(data_obj["edges"]), 2):
            edge1 = data_obj["edges"][i]
            if edge1[0] < index and edge1[1] < index:
                adj_matrix.append(edge1)
                edge_feats.append(data_obj["edge_attrs"][i])
                edge_pos.append([node_pos[int(edge1[0])], node_pos[int(edge1[1])]])

        # goal image and distances
        labels = []
        goal_feat = np.asarray(data_obj["nodes"]["feat"][goal_index])
        for ind in range(0, index):
            if self.args.distance_out:
                dist_score = self.geo_dist(ind, goal_index, data_obj)
            else:
                dist_score = (
                    1 - min(self.geo_dist(ind, goal_index, data_obj) ** 2, 10.0) / 10.0
                )
            labels.append(dist_score)
        #This is here in case we are using [0-1] predictions and need to get soft topk
        labelsDistance = []
        for ind in range(0, index):
            dist_score = self.geo_dist(ind, goal_index, data_obj)
            labelsDistance.append(dist_score)

        # valid_point
        pred_feat = []
        for v in range(0, len(valid_points)):
            valid_point = valid_points[v]
            new_index = index + v
            prev_index = int(valid_point[0])
            prev_pos = data_obj["nodes"]["pos"][prev_index]
            node_ids = np.append(node_ids, new_index)
            temp_pos = np.expand_dims(valid_point[1][0:3, 3], axis=0)
            node_pos = np.append(node_pos, temp_pos, axis=0)
            node_feat = np.append(node_feat, np.zeros((1, self.feat_size)), axis=0)
            edge_feats.append(valid_point[2])
            adj_matrix.append([prev_index, new_index])
            edge_pos.append([prev_pos, valid_point[1][0:3, 3]])
            if self.args.distance_out:
                valid_geo = self.geo_dist_valid(index, goal_index, node_pos, data_obj)
            else:
                valid_geo = self.geo_dist_valid(index, goal_index, node_pos, data_obj)
                valid_geo = 1 - min(valid_geo ** 2, 10.0) / 10.0
            labeldist_geo = self.geo_dist_valid(index, goal_index, node_pos, data_obj)

            labels.append(valid_geo)
            labelsDistance.append(labeldist_geo)
            pred_feat.append(new_index)
        pred_feat_dist_d = np.array(labels)[pred_feat]
        if self.args.clean_data:
            if len(np.where(pred_feat_dist_d == pred_feat_dist_d[0])[0]) == len(pred_feat_dist_d):
                return None

        adj_matrix = self.create_adj_matrix(adj_matrix)
        edge_feats = torch.flatten(
            torch.tensor(edge_feats, dtype=torch.float), start_dim=1
        )
        edge_pos = torch.tensor(edge_pos)

        # create torch geometric object
        goal_feat = np.expand_dims(goal_feat,1).repeat(node_feat.shape[0],1).T
        geo_data = Data(
            x=torch.tensor(node_feat, dtype=torch.float),
            edge_index=adj_matrix,
            edge_attr=edge_feats,
            edge_pos=edge_pos,
            pos=torch.tensor(node_pos, dtype=torch.float),
            goal_feat=torch.tensor(goal_feat, dtype=torch.float),
            pred_feat=torch.tensor(pred_feat),
            y=torch.tensor(labels),
            ydist=torch.tensor(labelsDistance),
            num_nodes=len(node_ids),
        )
        met_distances = {}
        for pos1 in geo_data.pos[pred_feat]:
            pos1 = pos1.tolist()
            for pos2 in geo_data.pos[pred_feat]:
                pos2 = pos2.tolist()
                key = (tuple(pos1),tuple(pos2))
                if key not in met_distances:
                    if pos1 == pos2:
                        distance = float(0)
                    else:
                        distance = self.sim.geodesic_distance(pos1, pos2)
                    met_distances[key] = distance
        geo_data.met_distances = met_distances

        return geo_data


    def load_node_info(self, data):
        data_list = []

        for d in tqdm(data):
            for data_obj in d:
                if len(data_obj["nodes"]["ids"]) < 10:
                    continue
                # graph starts with at least 2 nodes
                start_node_inx = 0
                end = len(data_obj["nodes"]["ids"]) - 1
                for i in range(start_node_inx + 1, end, 1):
                    length = len(data_obj["valid_points"][str(i - 1)])
                    if length > 1:
                        last_valid = data_obj["valid_points"][str(i - 1)]
                    else:
                        continue
                    # goal index can be anywhere between one behind to anywhere infront
                    goal_index = np.random.randint(
                        max(i - 2, start_node_inx + 1), min(i + 5, end + 1)
                    )
                    geo_data = self.true_node_to_point(
                        i, last_valid, goal_index, data_obj
                    )
                    if geo_data is not None:
                        data_list.append(geo_data)
        return data_list

    def create_adj_matrix(self, edge_list):
        edge_index = torch.tensor(edge_list, dtype=torch.long)
        edge_index = edge_index.t().contiguous()
        return edge_index

    def get_geo_dict(self, dict_location="./geo_dict.pkl"):
        path = Path(dict_location)
        if path.is_file():
            f = open(dict_location, 'rb')
            pickle.load(f)
        else:
            return {}

    def save_geo_dict(self, dict_location="./geo_dict.pkl"):
        f = open(dict_location, 'wb')
        pickle.dump(self.geo_dict, f)

    def get_sim(self, house):
        sim_dir = "../../data/scene_datasets/gibson_train_val"
        scene = "{}/{}.glb".format(sim_dir, house)
        sim, _ = set_up_habitat(scene)
        return sim

    def get_geo_dist(self,sim, pointA, pointB):
        geo_dist = sim.geodesic_distance(pointA, pointB)

    def build_dataset(self, split):
        splitFile = self.args.data_splits + "scenes_" + split + ".txt"
        splitScans = [x.strip() for x in open(splitFile, "r").readlines()]
        data_list = []
        if self.args.debug:
            splitScans = splitScans[0:1]

        for house in tqdm(splitScans):
            data = []
            self.sim = self.get_sim(house)
            houseFile = self.args.clustered_graph_dir + house + "_graphs.msg"
            data_temp = msgpack_numpy.unpack(open(houseFile, "rb"), raw=False)
            if self.args.debug:
                data_temp = data_temp[0:211]
            else:
                data_temp = data_temp[0:211]
            print(house)
            print(len(data_temp))
            data.append(data_temp)
            #Build dataset for triangle ineq
            data_list_temp = self.load_node_info([data_temp])
            self.sim.close()
            if self.args.debug:
                data_list.extend(data_list_temp)
            else:
                data_list.extend(data_list_temp)

        data_size = len(data_list)
        if self.args.expand_dataset and split=="train":
            data_list_symmetry = deepcopy(data_list)
            data_list_symmetry = self.symmetry_dataset(data_list_symmetry)
            data_list_identity = deepcopy(data_list)
            data_list_identity = self.identity_dataset(data_list_identity)
            data_list = data_list + data_list_symmetry + data_list_identity

        self.datasets[split] = Mp3dDataset(data_list)
        print("[{}]: Using {} top maps".format("data", data_size))
    def symmetry_dataset(self, data_list):
        #make 1/3 dataset symmetry
        for dl in data_list:
            x = dl.x
            dl.x = dl.goal_feat
            dl.goal_feat = x
        return data_list

    def identity_dataset(self, data_list):
        for dl in data_list:
            if random.random() > 0.5:
                dl.x = dl.goal_feat
            else:
                dl.goal_feat = dl.x
            dl.y = torch.zeros_like(dl.y)
        return data_list


class Mp3dDataset(Data):
    def __init__(
        self,
        data_list,
    ):
        self.data_list = data_list

    def __getitem__(self, index):
        data = self.data_list[index]
#        data = self.data_list[0]
        return data

    def __len__(self):
#        return 1
        return len(self.data_list)
