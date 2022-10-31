import argparse
import sys

parser = argparse.ArgumentParser(description="Image Nav Task")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dataset", type=str, default="gibson")
parser.add_argument("--tag", type=str, default="tag")
parser.add_argument(
    "--path_type", type=str, default="straight", help="options: [straight, curved]"
)
parser.add_argument(
    "--difficulty", type=str, default="hard", help="options: [easy, medium, hard]"
)

parser.add_argument("--visualize", default=False, action="store_true")
parser.add_argument("--visualization_dir", type=str, default="visualizations/")
parser.add_argument("--topdown_only", default=False, action="store_true")
parser.add_argument("--vis_as_graph", default=False, action="store_true")

# Baselines
parser.add_argument("--behavioral_cloning", default=False, action="store_true")
parser.add_argument("--straight_right_only", default=False, action="store_true")
parser.add_argument(
    "--bc_type", type=str, default="gru", help="options: [gru, map, random]"
)
parser.add_argument("--dont_reuse_poses", default=False, action="store_true")

# Abalations/GT
parser.add_argument("--double_check_switch",
                    default=False, action="store_true")
parser.add_argument("--use_gt_distances", default=False, action="store_true")
parser.add_argument("--use_gt_exporable_area",
                    default=False, action="store_true")
parser.add_argument("--use_gt_rhophi", default=False, action="store_true")
parser.add_argument("--use_pnp_rhophi", default=False, action="store_true")
parser.add_argument("--use_glue_rhophi", default=False, action="store_true")
parser.add_argument("--use_loftr_rhophi", default=False, action="store_true")
parser.add_argument("--use_gt_sim", default=False, action="store_true")
parser.add_argument("--use_rot", default=False, action="store_true")
parser.add_argument("--no_tc", default=False, action="store_true")
parser.add_argument("--no_switch", default=False, action="store_true")
parser.add_argument("--nrns_switch", default=False, action="store_true")
parser.add_argument("--model", type=str, default="TopoGCN")
parser.add_argument("--test_gt", type=str, default="TopoGCN")
parser.add_argument("--number_of_matches", type=int, default=10)
parser.add_argument("--switch_threshold", type=float, default=0.55)


# NOISE
parser.add_argument("--pose_noise", default=False, action="store_true")
parser.add_argument("--actuation_noise", default=False, action="store_true")

# RANDOM
parser.add_argument("--sample_used", type=float, default=1.0)
parser.add_argument("--max_steps", type=int, default=500)

# Data/Input Paths
parser.add_argument("--base_dir", type=str, default="../../data/topo_nav/")
parser.add_argument("--sim_dir", type=str,
                    default="../../data/scene_datasets/")
parser.add_argument("--floorplan_dir", type=str,
                    default="../../data/mp3d_floorplans/")

# Models
parser.add_argument("--model_dir", type=str, default="../../models/")
parser.add_argument("--distance_out", default=False, action="store_true")

parser.add_argument(
    "--distance_model_path",
    type=str,
    default="distance_gcn.pt",
    # NO NOISE or NOISY
    help="options: [distance_gcn.pt, distance_gcn_noise.pt, distance_BaselineOurs.pt]",
)

parser.add_argument(
    "--goal_model_path",
    type=str,
    default="goal_mlp.pt",
    # NO NOISE or NOISY
    help="options: [goal_mlp.pt, goal_mlp_noise.pt, goal_mlpBaselineOurs.pt]",
)

parser.add_argument(
    "--switch_model_path",
    type=str,
    default="switch_mlp.pt",
    help="options: [switch_mlp.pt, switch_mlp_noise.pt]",  # NO NOISE or NOISY
)

parser.add_argument(
    "--bc_model_path",
    type=str,
    default="bc_gru.pt",
    # (Metric Map + GRU + weighting) or (ResNet + Prev action + GRU + weighting)
    help="options: [bc_gru.pt, bc_metric_map.pt]",
)


def parse_args():
    if len(sys.argv) == 2:
        temp = [sys.argv[0]]
        temp.extend(sys.argv[1].split(" "))
        sys.argv = temp
    args = parser.parse_args()
    args.base_dir += f"{args.dataset}/"
    args.test_dir = f"{args.base_dir}image_nav_episodes/"

    if args.dataset == "mp3d":
        args.sim_dir += "mp3d/"
        args.bc_model_path = f"mp3d/mp3d_{args.bc_model_path}"
        args.switch_model_path = f"mp3d/mp3d_{args.switch_model_path}"
        args.goal_model_path = f"mp3d/mp3d_{args.goal_model_path}"
        args.distance_model_path = f"mp3d/mp3d_{args.distance_model_path}"
    else:
        args.sim_dir += "gibson_train_val/"
        args.bc_model_path = f"gibson/gibson_{args.bc_model_path}"
        args.switch_model_path = f"gibson/gibson_{args.switch_model_path}"
        args.goal_model_path = f"gibson/gibson_{args.goal_model_path}"
        args.distance_model_path = f"gibson/gibson_{args.distance_model_path}"
    return args
