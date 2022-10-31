import torch
import datetime
import random
from copy import deepcopy
import pudb
import numpy as np
import quaternion
import networkx as nx
from torch_geometric.data import Data
from habitat_sim import ShortestPath
from src.utils.sim_utils import diff_rotation


def gather_graph(agent):
    node_list = [n for n in agent.graph]
    adj = [list(e) for e in agent.graph.edges]
    for i in range(len(adj)):
        adj[i][0] = node_list.index(adj[i][0])
        adj[i][1] = node_list.index(adj[i][1])
    adj = torch.tensor(adj).T
    edge_attr = torch.stack([e[2]["attr"] for e in agent.graph.edges.data()])

    origin_nodes = []
    edge_rot = []
    unexplored_nodes = [
        n[0] for n in agent.graph.nodes.data() if n[1]["status"] == "unexplored"
    ]
    for e in agent.graph.edges.data():
        if e[1] in unexplored_nodes:
            origin_nodes.append(node_list.index(e[0]))
            edge_rot.append(e[2]["rotation"])

    unexplored_indexs = []
    explored_indexs = []
    for i, n in enumerate(agent.graph):
        if agent.graph.nodes.data()[n]["status"] == "unexplored":
            unexplored_indexs.append(i)
        else:
            explored_indexs.append(i)
    goal_feat = agent.goal_feat.repeat(len(node_list),1)
    geo_data = Data(
        goal_feat=goal_feat.clone().detach(),
        x=agent.node_feats.clone().detach()[node_list],
        edge_index=adj,
        edge_attr=edge_attr,
        ue_nodes=torch.tensor(unexplored_indexs),
        num_nodes=len(node_list),
    )
    return geo_data, unexplored_indexs, explored_indexs


def predict_distances(agent):
    if agent.use_gt_distances:
        pred_dists = gt_distances(agent)
        IGNORE_TRAVEL = agent.args.no_tc
        total_cost = add_travel_distance(agent, pred_dists, rot_thres=0.25, ignore_travel=IGNORE_TRAVEL)
        next_node = agent.unexplored_nodes[total_cost.argmin()]
        pred_dists = np.asarray(pred_dists)
        if total_cost[total_cost.argmin()] - pred_dists[pred_dists.argmin()] >= 10:
            next_node = agent.unexplored_nodes[pred_dists.argmin()]
        return next_node
    else:
        with torch.no_grad():
            IGNORE_TRAVEL = agent.args.no_tc
            agent_goal_pos = np.asarray(agent.goal_pos.tolist())
            geo_data, ue_nodes, e_nodes = gather_graph(agent)

            agent.gd_count += 1
            output = agent.feat_model([geo_data])[0].detach().cpu().squeeze(1)
            if agent.args.distance_out:
                pred_dists_dist_model = output[ue_nodes]
                total_cost_dist_model = add_travel_distance(agent, pred_dists_dist_model, rot_thres=0.25, ignore_travel=IGNORE_TRAVEL)
            if not agent.args.distance_out:
                pred_dists_dist_model = 10 * (1 - (output))[ue_nodes]
                total_cost_dist_model = add_travel_distance(agent, pred_dists_dist_model, rot_thres=0.25, ignore_travel=IGNORE_TRAVEL)
            next_node_dist_model = agent.unexplored_nodes[total_cost_dist_model.argmin()]

            output_old = deepcopy(output)
            #get gt
            pred_dists_gt = gt_distances(agent)
            total_cost_gt = add_travel_distance(agent, pred_dists_gt, rot_thres=0.25, ignore_travel=IGNORE_TRAVEL)
            next_node_gt = agent.unexplored_nodes[total_cost_gt.argmin()]
#            if next_node_gt != next_node_dist_model:
#                model_pred = total_cost_dist_model.argmin()
#                gt_short = total_cost_gt.min()
#                model_short = total_cost_gt[model_pred]
#                pu.db
            #get old output
            pred_dists_old = 10 * (1 - (output_old))[ue_nodes]
            total_cost_old = add_travel_distance(agent, pred_dists_old, rot_thres=0.25, ignore_travel=IGNORE_TRAVEL)
            next_node_old = agent.unexplored_nodes[total_cost_old.argmin()]
            RETURN = next_node_dist_model
            VIS_COST = total_cost_dist_model
            agent.distance_pred[len(agent.prev_poses)] = (deepcopy(agent.unexplored_nodes), deepcopy(VIS_COST), deepcopy(next_node_gt), deepcopy(agent.graph.edges))
            if not (RETURN == next_node_old):
                agent.node_switch_count += 1
            if not (RETURN == next_node_gt):
                agent.gt_switch_count += 1
    RETURN = next_node_dist_model
    return RETURN

def weight_nodes(ue_nodes, e_nodes, agent):
    closest_node_dist = []
    for node in ue_nodes:
        distance = np.inf
        for node_j in e_nodes:
            assert node != node_j
            d = torch.norm(agent.node_poses[node] - agent.node_poses[node_j]).item()
            #might need to check rot...
            if d == 0:
                distance = np.inf
                break
            if d < distance:
                distance = d
        closest_node_dist.append(distance)
    exp_distances = [np.exp(-i) for i in closest_node_dist]
    total = np.sum(exp_distances)
    exp_distances_norm = np.array([1 - (i/total) for i in exp_distances])
    inf_mask = np.where(np.array(closest_node_dist) == np.inf)[0] 
    exp_distances_norm[inf_mask] = np.inf
    return exp_distances_norm.tolist()


def add_travel_distance(agent, pred_dists, rot_thres, ignore_travel=True):
    #This is for scripting...
    if ignore_travel:
        return np.asarray(pred_dists)
    total_cost = []
    for n, goal_dist in zip(agent.unexplored_nodes, pred_dists):
        travel_dist = 0.25 * len(
            nx.shortest_path(agent.graph, source=agent.current_node, target=n)
        )
        quat1 = quaternion.from_float_array(agent.current_rot)
        quat2 = quaternion.from_float_array(agent.node_rots[n])
        rot_diff = rot_thres * diff_rotation(quat1, quat2) / 15
        #travel_dist = agent -> node
        #goal_dist = node -> goal
        total_cost.append(travel_dist + goal_dist + rot_diff)
    return np.asarray(total_cost)

def gt_distances(agent, nodes=None):
    if nodes == None:
        nodes = agent.unexplored_nodes
    distances = []
    for node_id in nodes:
        path = ShortestPath()
        path.requested_start = agent.node_poses[node_id].numpy().tolist()
        path.requested_end = agent.goal_pos.tolist()
        agent.pathfinder.find_path(path)
        distances += [path.geodesic_distance]
    return distances
