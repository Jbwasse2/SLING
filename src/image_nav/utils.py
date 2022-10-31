import time
import random
import os
import torch
import torch.nn as nn
import numpy as np
from torch_geometric.nn import DataParallel, GCNConv, CGConv
from src.functions.target_func.switch_model import SwitchMLP
from src.functions.target_func.goal_model import GoalMLP
from src.functions.feat_pred_fuc.deepgcn import TopoGCN
from src.functions.feat_pred_fuc.deepgcnNRNS import TopoGCNNRNS
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import poll_checkpoint_folder
from habitat_baselines.config.default import get_config
from habitat_baselines.utils.env_utils import construct_envs
from models.matching import Matching
from loftr.loftr import LoFTR, default_cfg
from copy import deepcopy

"""Evaluate Episode"""


def get_ddppo_model(config, device):
    def get_checkpoint(config):
        current_ckpt = None
        prev_ckpt_ind = -1
        while current_ckpt is None:
            current_ckpt = poll_checkpoint_folder(
                config.EVAL_CKPT_PATH_DIR, prev_ckpt_ind)
            time.sleep(0.1)  # sleep for 2 secs before polling again
        return current_ckpt
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


def evaluate_episode(agent, args, length_shortest):
    success = []
    dist_thres = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    dist_to_goal = np.linalg.norm(
        np.asarray(agent.goal_pos.tolist()) -
        np.asarray(agent.current_pos.tolist())
    )
    for thres in dist_thres:
        if dist_to_goal <= thres and agent.steps < args.max_steps:
            success.append(True)
        else:
            success.append(False)
    episode_spl = calculate_spl(success, length_shortest, agent.length_taken)
    return dist_to_goal, episode_spl, success


def calculate_spl(success, length_shortest, length_taken):
    spl = (length_shortest * 1.0 / max(length_shortest, length_taken))
    spl = list(map(lambda x: x * spl, success))
    return spl


"""Load Models"""


def load_models(args):
    """Action Pred function"""
    if args.bc_type == "map":
        from src.functions.bc_func.bc_map_network import ActionNetwork
    else:
        from src.functions.bc_func.bc_gru_network import ActionNetwork
    model_action = ActionNetwork()
    model_action.load_state_dict(torch.load(
        args.model_dir + args.bc_model_path))
    model_action.to(args.device)
    model_action.eval()

    """Load Switch function"""
    model_switch = SwitchMLP()
    model_switch.load_state_dict(torch.load(
        args.model_dir + args.switch_model_path))
    print(sum(p.numel() for p in model_switch.parameters()))
    model_switch.to(args.device)
    model_switch.eval()

    """Load Target function"""
    model_goal = GoalMLP()
    model_goal.load_state_dict(torch.load(
        args.model_dir + args.goal_model_path))
    model_goal.to(args.device)
    model_goal.eval()

    """Load Distance function"""
    if args.model == "TopoGCN":
        model_feat_pred = TopoGCN()
        #print(torch.sum(model_feat_pred.state_dict()['conv1.att.0.weight']))
    elif args.model == "TopoGCNNRNS":
        model_feat_pred = TopoGCNNRNS()
#        print(torch.sum(model_feat_pred.state_dict()['conv3.lin_f.weight']))
    model_feat_pred_w = torch.load(args.model_dir + args.distance_model_path)
#    print(torch.sum(model_feat_pred_w.state_dict()['conv3.lin.weight']))
    if isinstance(model_feat_pred_w, dict):
        model_feat_pred.load_state_dict(model_feat_pred_w['state_dict'])
        model_feat_pred = DataParallel(model_feat_pred)
    else:
        model_feat_pred = model_feat_pred_w

#    if args.model == "TopoGCN":
#        print(torch.sum(model_feat_pred.state_dict()
#              ['module.conv1.att.0.weight']))
#    elif args.model == "TopoGCNNRNS":
#        print(torch.sum(model_feat_pred.state_dict()
#              ['module.conv1.lin_f.weight']))

    model_feat_pred.to(args.device)
    model_feat_pred.eval()

    """DDPPO Model"""
#    print("DDPPO IS FOR GIBSON")
#    config = get_config("configs/baselines/ddppo_pointnav.yaml", [])
#    ddppo, ddppo_hidden = get_ddppo_model(config, args.device)
    ddppo, ddppo_hidden = None, None

    """superGlue"""
    config = {
        'superpoint': {
            'nms_radius': 4,
            'keypoint_threshold': 0.005,
            'max_keypoints': 1024
        },
        'superglue': {
            'weights': 'indoor',
            'sinkhorn_iterations': 20,
            'match_threshold': 0.2,
        }
    }
    matching = Matching(config).eval().to(args.device)

    _default_cfg = deepcopy(default_cfg)
    # set to False when using the old ckpt
    _default_cfg['coarse']['temp_bug_fix'] = True
    matcher = LoFTR(config=_default_cfg)
    matcher.load_state_dict(torch.load(
        "weights/indoor_ds_new.ckpt")['state_dict'])
    matcher = matcher.eval().cuda()

    return model_switch, model_goal, model_feat_pred, model_action, ddppo, ddppo_hidden, matching, matcher
