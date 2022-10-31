import gzip
import torch.nn.functional as F
from glob import glob
import os
import random
import time
from copy import deepcopy
from typing import ClassVar, Dict, List

import cv2
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pudb
import torch
import torchvision.transforms as transforms
from quaternion import as_euler_angles, quaternion
from tqdm import tqdm

import pudb

import habitat
from data.results.sparsifier.best_model.model import Siamese
from habitat import Config, logger
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import (
    agent_state_target2ref,
    angle_between_quaternions,
    quaternion_rotate_vector,
)

from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import poll_checkpoint_folder
from habitat_baselines.utils.env_utils import construct_envs

set_GPU = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = set_GPU


def transform_image(image):
    trnsform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    image = cv2.resize(image, (224, 224)) / 255
    image = trnsform(image)
    return image


def get_node_image_sequence(node, scene, max_lengths, transform=False, context=10):
    ret = []
    trnsform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    # Probably wanna cache this later...
    if node[0] not in max_lengths:
        max_length = len(
            glob(
                (
                    "../../data/datasets/pointnav/gibson/v4/train_large/images/"
                    + scene
                    + "/"
                    + "episodeRGB"
                    + str(node[0])
                    + "_*.jpg"
                )
            )
        )
        max_lengths[node[0]] = max_length
    else:
        max_length = max_lengths[node[0]]
    for i in range(node[1] - context, node[1]):
        i = max(i, 0)
        i = min(i, max_length)
        image_location = (
            "../../data/datasets/pointnav/gibson/v4/train_large/images/"
            + scene
            + "/"
            + "episodeRGB"
            + str(node[0])
            + "_"
            + str(i).zfill(5)
            + ".jpg"
        )
        image = plt.imread(image_location)
        image = cv2.resize(image, (224, 224)) / 255
        if transform:
            ret.append(trnsform(image))
        else:
            ret.append(image)
    return torch.stack(ret), max_lengths


def get_dict(fname):
    f = gzip.open(
        "../../data/datasets/pointnav/gibson/v4/train_large/content/"
        + fname
        + ".json.gz"
    )
    content = f.read()
    content = content.decode()
    content = content.replace("null", "None")
    content = eval(content)
    return content["episodes"]


def get_checkpoint(config):
    current_ckpt = None
    prev_ckpt_ind = -1
    while current_ckpt is None:
        current_ckpt = poll_checkpoint_folder(config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind)
        time.sleep(0.1)  # sleep for 2 secs before polling again
    return current_ckpt


def get_ddppo_model(config, device):
    checkpoint = get_checkpoint(config)
    ckpt_dict = torch.load(checkpoint)
    trainer = PPOTrainer(config)
    config.defrost()
    config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
    config.freeze()
    ppo_cfg = config.RL.PPO
    trainer.envs = construct_envs(config, get_env_class(config.ENV_NAME))
    trainer.device = device
    trainer._setup_actor_critic_agent(ppo_cfg)
    trainer.agent.load_state_dict(ckpt_dict["state_dict"])
    actor_critic = trainer.agent.actor_critic
    actor_critic.eval()
    test_recurrent_hidden_states = torch.zeros(
        actor_critic.net.num_recurrent_layers,
        config.NUM_PROCESSES,
        ppo_cfg.hidden_size,
        device=device,
    )
    return actor_critic, test_recurrent_hidden_states


def create_sim(scene):
    cfg = habitat.get_config("../../configs/tasks/pointnav_gibson.yaml")
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR"]
    frame_multiplier = 1
    cfg.SIMULATOR.FORWARD_STEP_SIZE = 0.25 / frame_multiplier
    cfg.SIMULATOR.TURN_ANGLE = 10 / frame_multiplier
    cfg.SIMULATOR.TILT_ANGLE = 10 / frame_multiplier
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


def example_forward(model, hidden_state, scene, device):
    ob = scene.reset()
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    not_done_masks += 1
    ob["pointgoal_with_gps_compass"] = torch.rand(1, 2).to(device)
    ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
    model.act(ob, hidden_state, prev_action, not_done_masks, deterministic=False)


# Takes a node such as (0,5) and reutrns its heading/position in the collected trajectory
def get_node_pose(node, d):
    try:
        pose = d[node[0]][node[1]]
        position = pose["position"]
        rotation = pose["rotation"]
    except KeyError:
        try:
            pose = d[node[0]]["shortest_paths"][0][0][node[1]]
        except Exception as e:
            print("FUCK")
        position = pose["position"]
        rotation = pose["rotation"]
    return position, rotation


def try_to_reach(
    G,
    start_node,
    end_node,
    d,
    ddppo_model,
    localization_model,
    hidden_state,
    sim,
    device,
    visualize=True,
):
    ob = sim.reset()
    scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
    # Perform high level planning over graph
    if visualize:
        name = scene + "_" + str(start_node) + "_" + str(end_node)
        video_name = "./videos/" + name + ".mkv"
        video = cv2.VideoWriter(video_name, 0, 3, (256 * 2, 256 * 1))
    else:
        video = None
    try:
        path = nx.dijkstra_path(G, start_node, end_node)
    except nx.exception.NetworkXNoPath as e:
        return 3, -1
    print("NEW PATH")
    MAX_NUMBER_OF_STEPS = 500
    print("Length of Path is " + str(len(path)))
    print(MAX_NUMBER_OF_STEPS)
    number_of_steps_left = MAX_NUMBER_OF_STEPS * len(path)
    number_of_steps_left = MAX_NUMBER_OF_STEPS
    current_node = path[0]
    local_goal = path[1]
    # Move robot to starting position/heading
    agent_state = sim.agents[0].get_state()
    ground_truth_d = get_dict(scene)
    pos, rot = get_node_pose(current_node, ground_truth_d)
    agent_state.position = pos
    agent_state.rotation = rot
    sim.agents[0].set_state(agent_state)
    # Start experiments!
    for current_node, local_goal in zip(path, path[1:]):
        success, number_of_steps_left = try_to_reach_local(
            current_node,
            local_goal,
            d,
            ddppo_model,
            localization_model,
            hidden_state,
            sim,
            device,
            video,
            number_of_steps_left,
        )
        if success != 1:
            if visualize:
                cv2.destroyAllWindows()
                video.release()
            return 1, len(path)
    if visualize:
        cv2.destroyAllWindows()
        video.release()
    # Check to see if agent made it
    agent_pos = sim.agents[0].get_state().position
    ground_truth_d = get_dict(scene)
    (episode, frame, local_pose, global_pose) = end_node
    goal_pos = ground_truth_d[episode]["shortest_paths"][0][0][frame]["position"]
    distance = np.linalg.norm(agent_pos - goal_pos)
    if distance >= 0.4:
        print(distance)
        return 2, len(path)
    return 0, len(path)


def get_node_depth(node, scene_name):
    image_location = (
        "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        + scene_name
        + "/"
        + "episodeDepth"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    return plt.imread(image_location)


def get_node_image(node, scene_name):
    image_location = (
        "../../data/datasets/pointnav/gibson/v4/train_large/images/"
        + scene_name
        + "/"
        + "episodeRGB"
        + str(node[0])
        + "_"
        + str(node[1]).zfill(5)
        + ".jpg"
    )
    return plt.imread(image_location)


# Returns 1 on success, and 0 or -1 on failure
def try_to_reach_local(
    start_node,
    local_goal_node,
    d,
    ddppo_model,
    localization_model,
    hidden_state,
    sim,
    device,
    video,
    number_of_steps_left,
    context=9,
):
    # print(number_of_steps_left)
    prev_action = torch.zeros(1, 1).to(device)
    not_done_masks = torch.zeros(1, 1).to(device)
    not_done_masks += 1
    ob = sim.get_observations_at(sim.get_agent_state())
    if np.sum(ob["rgb"]) == 0.0:
        print("NO OBSERVATION")
        return 1
    actions = []
    # Double check this is right RGB
    # goal_image is for video/visualization
    # goal_image_model is for torch model for predicting distance/heading
    scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]
    max_lengths = {}
    scene = os.path.basename(sim.config.sim_cfg.scene.id).split(".")[0]
    local_images_buffer, max_lengths = get_node_image_sequence(
        start_node, scene, max_lengths, transform=True
    )
    goal_images_buffer, max_lengths = get_node_image_sequence(
        local_goal_node, scene, max_lengths, transform=True
    )
    if video is not None:
        scene_name = os.path.splitext(os.path.basename(sim.config.sim_cfg.scene.id))[0]

        goal_image = cv2.resize(get_node_image(local_goal_node, scene_name), (256, 256))
    try:
        for i in range(number_of_steps_left):
            displacement, prob = get_displacement_local_goal(
                local_images_buffer, goal_images_buffer, localization_model, device
            )
            #            if prob[0] <= 0.90:
            #                return 0, 0
            displacement = torch.from_numpy(displacement).type(torch.float32)
            # displacement = torch.from_numpy(
            #    get_displacement_local_goal_oracle(sim, local_goal_node, d)
            # ).type(torch.float32)

            if video is not None:
                image = np.hstack([ob["rgb"], goal_image])
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(
                    image,
                    text=str(displacement).replace("tensor(", "").replace(")", ""),
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    org=(10, 10),
                    fontScale=0.5,
                    color=(255, 255, 255),
                    thickness=2,
                )
                video.write(image)
            ob["pointgoal_with_gps_compass"] = displacement.unsqueeze(0).to(device)
            ob["depth"] = torch.from_numpy(ob["depth"]).unsqueeze(0).to(device)
            with torch.no_grad():
                _, action, _, hidden_state = ddppo_model.act(
                    ob, hidden_state, prev_action, not_done_masks, deterministic=True
                )
            actions.append(action[0].item())
            prev_action = action
            if action[0].item() == 0:  # This is stop action
                return 1, number_of_steps_left - 1 - i
            ob = sim.step(action[0].item())
            # add new observation to buffer
            for i in range(context):
                local_images_buffer[i] = local_images_buffer[i + 1]
            local_images_buffer[context] = transform_image(ob["rgb"])
    except Exception as e:
        print(e)
    return 0, 0


def get_displacement_local_goal(local_images_buffer, goal_images_buffer, model, device):
    local_images_buffer = local_images_buffer.to(device).float()
    goal_images_buffer = goal_images_buffer.to(device).float()
    if len(local_images_buffer.shape) == 4:
        local_images_buffer = local_images_buffer.unsqueeze(0)
    if len(goal_images_buffer.shape) == 4:
        goal_images_buffer = goal_images_buffer.unsqueeze(0)
    hidden = model.init_hidden(local_images_buffer.shape[0], model.hidden_size, device)
    model.hidden = hidden
    pose, prob = model(local_images_buffer, goal_images_buffer)
    pose = pose.detach().cpu().numpy()[0]
    prob = F.softmax(prob)
    print(prob)
    rho = pose[0]
    phi = pose[1]
    return np.array([np.abs(rho), phi]), prob


def get_displacement_local_goal_oracle(sim, local_goal, d):
    # for more information
    pos_goal, rot_goal = get_node_pose(local_goal, d)
    # Quaternion is returned as list, need to change datatype
    if isinstance(rot_goal, type(np.array)):
        rot_goal = quaternion(*rot_goal)
    pos_agent = sim.get_agent_state().position
    rot_agent = sim.get_agent_state().rotation
    direction_vector = pos_goal - pos_agent
    direction_vector_agent = quaternion_rotate_vector(
        rot_agent.inverse(), direction_vector
    )
    rho, phi = cartesian_to_polar(-direction_vector_agent[2], direction_vector_agent[0])

    # Should be same as agent_world_angle
    return np.array([rho, -phi])


def visualize_observation(observation, start, goal):
    f, ax = plt.subplots(1, 3)
    ax[0].imshow(observation["rgb"])
    ax[0].title.set_text("Current Observation")
    ax[1].imshow(start)
    ax[1].title.set_text("Local Start Image")
    ax[2].imshow(goal)
    ax[2].title.set_text("Local Goal Image")
    plt.show()


def run_experiment(
    G, d, ddppo_model, localization_model, hidden_state, scene, device, experiments=100
):
    # Choose 2 random nodes in graph
    # 0 means success
    # 1 means failed at runtime
    # 2 means the system thought it finished the path, but was actually far away
    # 3 means topological map failed to find a path
    return_codes = [0 for i in range(4)]
    length_results = []
    for _ in tqdm(range(experiments)):
        results = None
        node1, node2 = get_two_nodes(G)
        results, path_length = try_to_reach(
            G,
            node1,
            node2,
            d,
            ddppo_model,
            localization_model,
            deepcopy(hidden_state),
            scene,
            device,
        )
        length_results.append((node1[0:2], node2[0:2], path_length, results))
        return_codes[results] += 1
    print("Length Results = " + str(length_results))
    return return_codes


# Currently just return 2 random nodes, in the future may do something smarter.
def get_two_nodes(G):
    return random.sample(list(G.nodes()), 2)


def get_localization_model(device):
    model = Siamese().to(device)
    model.load_state_dict(
        torch.load(
            "./data/results/sparsifier/best_model/saved_model.pth",
        )
    )
    model.eval()
    return model


def ruin_map(G, percent_good):
    wormholes = 0
    gaps = 0
    G_edges = list(G.edges())
    for node1 in tqdm(list(G.nodes())):
        for node2 in list(G.nodes()):
            if node1 == node2:
                continue
            edge = (node1, node2)
            if np.random.uniform() > percent_good:
                if edge in G_edges:
                    if len(G.get_edge_data(*edge)) > 0:
                        gaps += 1
                        G.remove_edge(*edge)
                else:
                    G.add_edge(*edge)
                    wormholes += 1
    print("Number of Wormholes = " + str(wormholes))
    print("Number of Gaps = " + str(gaps))
    return G


# def make_sparse_map(G, d, distance):
#    nodes = list(G.nodes())
#    removed_nodes = []
#    for node1 in nodes:
#        for node2 in nodes:
#            if node1 == node2:
#                break
#            pose1 = d[node1[0]]["shortest_paths"][0][0][node1[1]]["position"]
#            pose2 = d[node2[0]]["shortest_paths"][0][0][node2[1]]["position"]
#            pu.db


def plot_G(G, G_og, fname, d):
    fig = plt.figure(figsize=(8, 8))

    def make_plot(G_l):
        G_nodes = list(G_l.nodes())
        G_edges = list(G_l.edges())
        x = []
        y = []
        z = []
        for node in tqdm(G_nodes):
            pos, rot = get_node_pose(node, d)
            x.append(pos[0])
            y.append(pos[1])
            z.append(pos[2])
        if np.max(y) - np.min(y) >= 0.5:
            print("Warning possible multiple levels!")
        ax.scatter(x, z, c=y, cmap="winter")
        for edge in tqdm(G_edges):
            node1, node2 = edge
            pos1, rot = get_node_pose(node1, d)
            pos2, rot = get_node_pose(node2, d)
            diff = np.array(pos2) - np.array(pos1)
            plt.arrow(
                pos1[0], pos1[2], diff[0], diff[2], color="Black", alpha=0.1, width=0.01
            )

    #            plt.plot(
    #                [pos1[0], pos2[0]],
    #                #                [pos1[1], pos2[1]],
    #                [pos1[2], pos2[2]],
    #                color="Black",
    #                alpha=0.1,
    #            )

    # First plot difference of graphs
    ax = fig.add_subplot(1, 1, 1)
    G_nodes = list(G.nodes())
    G_edges = list(G.edges())
    G_og_edges = list(G_og.edges())
    x = []
    y = []
    z = []
    for edge in tqdm(G_edges):
        if edge not in G_og_edges:
            node1, node2 = edge
            pos1, rot = get_node_pose(node1, d)
            pos2, rot = get_node_pose(node2, d)
            plt.plot(
                [pos1[0], pos2[0]],
                #                [pos1[1], pos2[1]],
                [pos1[2], pos2[2]],
                color="red",
                alpha=0.1,
            )
    for edge in tqdm(G_og_edges):
        if edge not in G_edges:
            node1, node2 = edge
            pos1, rot = get_node_pose(node1, d)
            pos2, rot = get_node_pose(node2, d)
            plt.plot(
                [pos1[0], pos2[0]],
                #                [pos1[1], pos2[1]],
                [pos1[2], pos2[2]],
                color="yellow",
                alpha=1.0,
            )
    for node in tqdm(G_nodes):
        pos, rot = get_node_pose(node, d)
        x.append(pos[0])
        y.append(pos[1])
        z.append(pos[2])
    ax.scatter(x, z, c=y, cmap="winter")
    plt.savefig(fname + "_diff.png")
    plt.clf()

    # Now plot graphs
    ax = fig.add_subplot(1, 1, 1)
    make_plot(G_og)
    plt.savefig(fname + "_OG.png")
    plt.clf()
    ax = fig.add_subplot(1, 1, 1)
    make_plot(G)
    plt.savefig(fname + "_G.png")
    plt.clf()


def get_prec_recall(G, G_og):
    tp, fp, fn = 0, 0, 0
    G_og_edges = list(G_og.edges())
    G_edges = list(G.edges())
    G_nodes = list(G.nodes())
    for node1 in tqdm(G_nodes):
        for node2 in G_nodes:
            if node1 == node2:
                continue
            gt = (node1, node2) in G_og_edges
            guess = (node1, node2) in G_edges
            if gt is True and guess is True:
                tp += 1
            elif gt is False and guess is True:
                fp += 1
            elif gt is True and guess is False:
                fn += 1
    return tp / (tp + fp), tp / (tp + fn)


def main():
    env = "Browntown"
    print(env)
    map_type_test = "perfect"
    print(map_type_test)
    seed = 0
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if map_type_test in [
        "topological",
        "similarity_orbslamRGB",
        "similarity_orbslamRGBD",
    ]:
        test_similarityEdges = 0.97
        closeness = 1.0
    elif map_type_test in ["perfect", "perfect2", "VO", "orbslamRGB", "orbslamRGBD"]:
        test_similarityEdges = None
        closeness = 1.0
        percent_good = 1.0
        dist_merge = 1.0
    elif map_type_test in ["similarity"]:
        test_similarityEdges = 0.99
        closeness = None
    elif map_type_test in ["base"]:
        test_similarityEdges = None
        closeness = 0.0
    else:
        assert 1 == 0
    G = nx.read_gpickle(
        "./data/map/"
        + str(map_type_test)
        + "/map50Worm20NewArchTest_"
        + str(env)
        + str(test_similarityEdges)
        + "_"
        + str(closeness)
        + ".gpickle",
    )
    d = get_dict(env)
    if map_type_test is "perfect" or map_type_test is "perfect2":
        G_og = deepcopy(G)
        #        G = make_sparse_map(G, d, dist_merge)
        G = ruin_map(G, percent_good)
        #    plot_G(
        #        G,
        #        G_og,
        #        "./results/visualize_3d/" + map_type_test + env + str(percent_good),
        #        d,
        #    )
        print("nodes = ", len(G.nodes()))
        print("edges = ", len(G.edges()))
        # print(get_prec_recall(G, G_og))
    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
    device = (
        torch.device("cuda", config.TORCH_GPU_ID)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    localization_model = get_localization_model(device)
    scene = create_sim(env)
    ddppo_model, hidden_state = get_ddppo_model(config, device)
    # example_forward(model, hidden_state, scene, device)
    # d = np.load("../data/map/d_slam.npy", allow_pickle=True).item()
    results = run_experiment(
        G, d, ddppo_model, localization_model, hidden_state, scene, device
    )
    print(results)


if __name__ == "__main__":
    main()
