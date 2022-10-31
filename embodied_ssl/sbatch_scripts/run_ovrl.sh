#!/bin/bash
source ./sbatch_scripts/training_function.sh

## Slurm
REPO_PATH=/path/to/code/embodied_ssl
BASE_TASK_CONFIG_PATH="${REPO_PATH}/configs/tasks/dino_imagenav_gibson.yaml"
EXP_CONFIG_PATH="${REPO_PATH}/ssl_baselines/config/imagenav/dino_ddppo_imagenav_gibson.yaml"
ENVIRONMENT="gibson"
VIDEO_OPTION="[]"
TEST_EPISODE_COUNT=3000

CONFIGS="RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.brightness 0.3 \
        RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.contrast 0.3 \
        RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.saturation 0.3 \
        RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.hue 0.3 \
        RL.POLICY.OBS_AUGMENTATIONS_PARAMS.COLOR_JITTER.color_p 1.0 \
        RL.POLICY.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.pad 4 \
        RL.POLICY.OBS_AUGMENTATIONS_PARAMS.TRANSLATE.crop_size 128\
        RL.POLICY.OBS_AUGMENTATIONS color_jitter-translate_v2 \
        RL.PPO.weight_decay 1e-6 \
        TASK_CONFIG.SIMULATOR.RGB_SENSOR.WIDTH 128 \
        TASK_CONFIG.SIMULATOR.RGB_SENSOR.HEIGHT 128 \
        TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT 128 \
        TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH 128 \
        TASK_CONFIG.SIMULATOR.TURN_ANGLE 30"

EXP_NAME="ovrl_best_run"
WEIGHTS_NAME="omnidata_DINO_02.pth"
BACKBONE="resnet50_gn"
EXTRA_CMDS="RL.DDPPO.pretrained_encoder True \
            RL.DDPPO.pretrained_goal_encoder True \
            RL.DDPPO.train_encoder True \
            RL.DDPPO.train_goal_encoder True \
            ${CONFIGS}"
METHOD="ovrl_sling"
CHKP_NAME="best_ovrl_ckpt.pth"

for VAL_SPLIT in "gibson_curved_nrns" "gibson_straight_nrns" "matterport_straight_nrns" "matterport_curved_nrns"
do
    for DIFFICULTY in "easy" "medium" "hard"
    do
        run_training 0
    done
done

