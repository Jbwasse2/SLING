import habitat

import slam_agents
from pathlib import Path
import matplotlib.pyplot as plt
from habitat.utils.visualizations.maps import get_topdown_map
import gzip
import numpy as np
import torch
from slam_agents import (
    AgentMono,
    ORBSLAM2MonoAgent,
    ORBSLAM2Agent,
    get_config,
    cfg_baseline,
    make_good_config_for_orbslam2,
)
import argparse
import os


def create_agent(scene, agent_type):
    config = habitat.get_config("./configs/tasks/pointnav_rgbd.yaml")
    agent_config = cfg_baseline()
    agent_config.defrost()
    config.defrost()
    config.ORBSLAM2 = agent_config.ORBSLAM2
    config.ORBSLAM2.SLAM_VOCAB_PATH = "../../data/ORBvoc.txt"
    config.ORBSLAM2.SLAM_SETTINGS_PATH = "../../data/mp3d3_small1k.yaml"
    # config.ORBSLAM2.SLAM_SETTINGS_PATH = "./data/mono.yaml"
    config.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    make_good_config_for_orbslam2(config)

    if agent_type == "blind":
        agent = BlindAgent(config.ORBSLAM2)
    elif agent_type == "orbslam2-rgbd":
        agent = ORBSLAM2Agent(config.ORBSLAM2)
    elif agent_type == "orbslam2-rgb-monod":
        agent = ORBSLAM2MonodepthAgent(config.ORBSLAM2)
    elif agent_type == "mono":
        agent = AgentMono(config.ORBSLAM2)
    else:
        raise ValueError(agent_type, "is unknown type of agent")
    return agent, config


def create_sim(scene, cfg):
    cfg.defrost()
    cfg.SIMULATOR.SCENE = "../../data/scene_datasets/gibson/" + scene + ".glb"
    cfg.freeze()
    sim = habitat.sims.make_sim("Sim-v0", config=cfg.SIMULATOR)
    return sim


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


# Note to self, try save directory with name that corresponds to the agent type.
def add_traj_to_SLAM(agent, env, rgb, depth):
    poses = []
    counter = 0
    skips = 0
    for im, d in zip(rgb, depth):
        observation = {}
        observation["rgb"] = im
        observation["depth"] = d
        if agent.update_internal_state(observation) == False:
            skips += 1
        counter += 1
        poses.append(agent.trajectory_history[-1])
    return agent.pose6D


def get_actual_top_down(sim, env):
    plt.imsave(
        "./top_down/top_down_" + str(env) + ".png",
        get_topdown_map(sim.pathfinder, 0.01),
    )


# agent = mono, blind, orbslam2-rgbd, orbslam2-rgb-monod
def get_slam_pose_labels(env, agent, sim, rgb=None, depth=None):
    assert rgb != None or depth != None
    agent, config = create_agent(env, agent)
    if sim == None:
        sim = create_sim(env, config)
    pose = add_traj_to_SLAM(agent, env, rgb, depth)
    return pose


if __name__ == "__main__":
    scene = "Browntown"
    # mono, blind, orbslam2-rgbd (works), orbslam2-rgb-monod
    #    get_slam_pose_labels(scene, 20, agent="mono")
    print(get_slam_pose_labels(scene, 3, agent="orbslam2-rgbd"))
