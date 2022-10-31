import torch
import subprocess
import sys
import torch_geometric
import datetime
import random
import os
import numpy as np
import tqdm
import json
import gzip
import quaternion
from habitat.utils.geometry_utils import quaternion_from_coeff
import habitat_sim
import habitat
from torch.utils.tensorboard import SummaryWriter

from src.utils.model_utils import load_places_resnet
from src.image_nav.visualize import Visualizer
from src.image_nav.cfg import parse_args
from src.image_nav.agent import Agent
from src.image_nav.navigate import single_image_nav, single_image_nav_forward_right
from src.image_nav.bcloning_navigate import single_image_nav_BC
from src.image_nav.utils import load_models, evaluate_episode
from src.utils.sim_utils import (
    set_up_habitat_noise,
    set_up_habitat,
    add_noise_actions_habitat,
)

args = parse_args()
INSTANCE = 0
ins = INSTANCE
curr_time = "None"
save_path_global = "../../results/masterNav/" + args.tag + "/" + curr_time + "/"
if ins == 0:
    writer = SummaryWriter(save_path_global)
    # Add git stuff to folder
    os.system("git rev-parse HEAD > " + save_path_global + "gitversion")
    os.system("git diff > " + save_path_global + "gitdiff")
    os.system("unlink last_run")
    os.system("ln -sf " + str(save_path_global) + " last_run")
    f = open(save_path_global + "args", "w")
    f.write(str(args))
    f.close()
    f = open(save_path_global + "command", "w")
    f.write(str(subprocess.list2cmdline(sys.argv)))
    f.close()


turn_angle = 15
DUMP_JSON = False


def create_habitat(args, sim, current_scan):
    if sim is not None:
        sim.close()
    if args.dataset == "mp3d":
        scene = "{}{}/{}.glb".format(args.sim_dir, current_scan, current_scan)
    else:
        scene = "{}{}.glb".format(args.sim_dir, current_scan)

    return set_up_habitat_noise(scene, turn_angle)


def seed_everything(seed: int):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(args):
    seed_everything(args.seed)
    print(habitat.__version__)
    print(habitat_sim.__version__)
    print(f"starting run: {args.path_type} data")
    print("Data Type:", args.difficulty)
    print(
        f"Pose Noise: {args.pose_noise}; Actuation Noise: {args.actuation_noise}")
    """Evaluation Metrics"""
    maxed_out = 0
    rates = {
        "success0.4": [],
        "success0.5": [],
        "success0.6": [],
        "success0.7": [],
        "success0.8": [],
        "success0.9": [],
        "success1.0": [],
        "spl0.4": [],
        "spl0.5": [],
        "spl0.6": [],
        "spl0.7": [],
        "spl0.8": [],
        "spl0.9": [],
        "spl1.0": [],
        "dist2goal": [],
        "taken_path_total": [],
        "taken_path_success": [],
        "gt_path_total": [],
        "gt_path_success": [],
        "gt_count": [],
        "gea_count": [],
        "gd_count": [],
        "node_switch_count": [],
        "gt_switch_count": [],
        "steps": [],
        "episode_id": [],
    }
    rates["args"] = str(args)
    visCounter = {}
    visualizer = Visualizer(args)

    """Load Models"""
    args.device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu")
    resnet = load_places_resnet()
    model_switch, model_goal, model_feat_pred, model_action, ddppo, ddppo_hidden, super_glue, loftr = load_models(
        args)
    print("finished loading models.")

    """Create Habitat Sim"""
    add_noise_actions_habitat()
    sim, pathfinder, current_scan = None, None, None

    """Load test episodes"""
    jsonfilename = f"{args.test_dir}/{args.path_type}_{args.difficulty}.json.gz"
    print(jsonfilename)
    with gzip.open(jsonfilename, "r") as fin:
        data = json.loads(fin.read().decode("utf-8"))["episodes"]
    """Loop over test episodes"""
    INSTANCE = 0
    TOTAL_INSTANCES = 1
    start = int((1 / TOTAL_INSTANCES) * (INSTANCE) * len(data))
    end = int((1 / TOTAL_INSTANCES) * (INSTANCE + 1) * len(data))
    print(INSTANCE)
    print(start, end)
    json_f = save_path_global + str(INSTANCE) + ".json"
    data = data[start:end]
#    data = data[1:2]
    # Elmira_
    #data = data[261:262]
    # Cantwell_17
    #data = data[21:22]
    # Denmark 107
    #data = data[100:101]
    # Cantwell 81
#    data = data[41:42]

    for instance in tqdm.tqdm(data):
        scan_name = instance["scene_id"].split("/")[-1].split(".")[0]
        episode_id = instance["episode_id"]
        rates["episode_id"].append(episode_id)
        length_shortest = instance["length_shortest"]

        """Load habitat scene"""
        if current_scan != scan_name:
            current_scan = scan_name
            print(current_scan)
            sim, pathfinder = create_habitat(args, sim, current_scan)
            visCounter[current_scan] = 0

        """ Image nav agent per episode"""
        agent = Agent(
            args,
            sim,
            pathfinder,
            resnet,
            current_scan,
            model_goal,
            model_switch,
            model_feat_pred,
            model_action,
            ddppo,
            ddppo_hidden,
            super_glue,
            loftr

        )
        agent.scene_id = instance['episode_id']
        start_position = instance["start_position"]
        start_rotation = quaternion.as_float_array(
            quaternion_from_coeff(instance["start_rotation"])
        )
        goal_position = instance["goals"][0]["position"]
        goal_rotation = quaternion.as_float_array(
            quaternion_from_coeff(instance["goals"][0]["rotation"])
        )
        agent.reset_agent(start_position, start_rotation,
                          goal_position, goal_rotation)

        """ Run a image nav episode"""
        if args.behavioral_cloning:
            single_image_nav_BC(agent, args)
        elif args.straight_right_only:
            print("RUNNING STRAIGHT RIGHT")
            print("RUNNING STRAIGHT RIGHT")
            print("RUNNING STRAIGHT RIGHT")
            print("RUNNING STRAIGHT RIGHT")
            single_image_nav_forward_right(agent, visualizer, args)
        else:
            single_image_nav(agent, visualizer, args)

        """ Evaluate result of episode"""
        dist_to_goal, episode_spl, success = evaluate_episode(
            agent, args, length_shortest
        )
        if agent.steps >= args.max_steps:
            maxed_out += 1
        rates["success0.4"].append(success[0])
        rates["success0.5"].append(success[1])
        rates["success0.6"].append(success[2])
        rates["success0.7"].append(success[3])
        rates["success0.8"].append(success[4])
        rates["success0.9"].append(success[5])
        rates["success1.0"].append(success[6])
        rates["spl0.4"].append(episode_spl[0])
        rates["spl0.5"].append(episode_spl[1])
        rates["spl0.6"].append(episode_spl[2])
        rates["spl0.7"].append(episode_spl[3])
        rates["spl0.8"].append(episode_spl[4])
        rates["spl0.9"].append(episode_spl[5])
        rates["spl1.0"].append(episode_spl[6])
        rates["dist2goal"].append(dist_to_goal)
        rates["taken_path_total"].append(agent.length_taken)
        rates["gt_path_total"].append(length_shortest)
        rates["steps"].append(agent.steps)
        rates["gt_count"].append(agent.gt_count)
        rates["gea_count"].append(agent.gea_count)
        rates["gd_count"].append(agent.gd_count)
        rates["node_switch_count"].append(agent.node_switch_count)
        rates["gt_switch_count"].append(agent.gt_switch_count)

        if success[6]:
            rates["taken_path_success"].append(agent.length_taken)
            rates["gt_path_success"].append(length_shortest)
        else:
            rates["taken_path_success"].append(-1)
            rates["gt_path_success"].append(-1)

        """Visualize Episode"""
        if args.visualize:
            print("Creating visualization of episode...")
            visCounter[current_scan] += 1
            if args.dataset == "mp3d":
                visualizer.create_layout_mp3d(episode_id)
            else:
                visualizer.create_layout(agent, episode_id, success[6])
        print(np.mean(rates['success1.0']))
        print(np.mean(rates['spl1.0']))
        if DUMP_JSON:
            json_dump(agent, rates, json_f)

    """Print Stats"""
    print("\nType of Run: ")
    print(f"Dataset: {args.dataset.upper()}")
    print(f"Data: {args.path_type.upper()}")
    print(f"Data Type: {args.difficulty}")
    print(
        f"Pose Noise: {args.pose_noise}; Actuation Noise: {args.actuation_noise}")

    print("\nStats of Runs: ")
    print(f"Success Rate0.4: {np.mean(rates['success0.4']):.4f}")
    print(f"Success Rate0.5: {np.mean(rates['success0.5']):.4f}")
    print(f"Success Rate0.6: {np.mean(rates['success0.6']):.4f}")
    print(f"Success Rate0.7: {np.mean(rates['success0.7']):.4f}")
    print(f"Success Rate0.8: {np.mean(rates['success0.8']):.4f}")
    print(f"Success Rate0.9: {np.mean(rates['success0.9']):.4f}")
    print(f"Success Rate1.0: {np.mean(rates['success1.0']):.4f}")
    print(f"SPL0.4: {np.mean(rates['spl0.4']):.4f}")
    print(f"SPL0.5: {np.mean(rates['spl0.5']):.4f}")
    print(f"SPL0.6: {np.mean(rates['spl0.6']):.4f}")
    print(f"SPL0.7: {np.mean(rates['spl0.7']):.4f}")
    print(f"SPL0.8: {np.mean(rates['spl0.8']):.4f}")
    print(f"SPL0.9: {np.mean(rates['spl0.9']):.4f}")
    print(f"SPL1.0: {np.mean(rates['spl1.0']):.4f}")
    print(f"Avg dist to goal: {np.mean(rates['dist2goal']):.4f}")
    print(
        f"Avg taken path len - total: {np.mean(rates['taken_path_total']):.4f}")
    print(
        f"Avg taken path len - success: {np.mean(rates['taken_path_success']):.4f}")
    print(f"Avg gt path len - total: {np.mean(rates['gt_path_total']):.4f}")
    print(
        f"Avg gt path len - success: {np.mean(rates['gt_path_success']):.4f}")

    print("\nFor excel in above order: ")
    print(f"{np.mean(rates['success1.0']):.4f}")
    print(f"{np.mean(rates['spl1.0']):.4f}")
    print(f"{np.mean(rates['dist2goal']):.4f}")
    print(f"{np.mean(rates['taken_path_total']):.4f}")
    print(f"{np.mean(rates['taken_path_success']):.4f}")
    print(f"{np.mean(rates['gt_path_total']):.4f}")
    print(f"{np.mean(rates['gt_path_success']):.4f}")
    print(np.sum(rates['node_switch_count']))
    print(np.sum(rates['gt_count']))
    print(np.sum(rates['gea_count']))
    print(np.sum(rates['gd_count']))


def json_dump(agent, rates, f):
    with open(f, 'w', encoding='utf-8') as f:
        json.dump(rates, f)


if __name__ == "__main__":
    main(args)
