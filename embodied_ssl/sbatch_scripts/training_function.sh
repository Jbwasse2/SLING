run_training() {
    SEED=$1
    # create run folder
    RUN_FOLDER="${REPO_PATH}/checkpoint/${EXP_NAME}"
    LOG_DIR="${RUN_FOLDER}/logs_${VAL_SPLIT}_${DIFFICULTY}_${METHOD}"
    CHKP_DIR="${RUN_FOLDER}/chkp"
    VIDEO_DIR="${RUN_FOLDER}/videos"

    # Create folders
    mkdir -p ${CHKP_DIR}
    mkdir -p ${LOG_DIR}
    mkdir -p ${VIDEO_DIR}

    if [ -z "${CHKP_NAME}" ]; then
        EVAL_CKPT_PATH_DIR="${CHKP_DIR}"
    else
        EVAL_CKPT_PATH_DIR="${CHKP_DIR}/${CHKP_NAME}"
    fi

    # Write commands to file
    CMD_COMMON_OPTS="--exp-config $EXP_CONFIG_PATH \
        BASE_TASK_CONFIG_PATH $BASE_TASK_CONFIG_PATH \
        EVAL_CKPT_PATH_DIR ${EVAL_CKPT_PATH_DIR} \
        CHECKPOINT_FOLDER ${CHKP_DIR} \
        TENSORBOARD_DIR ${LOG_DIR} \
        VIDEO_DIR ${VIDEO_DIR} \
        RL.DDPPO.pretrained_weights ${REPO_PATH}/data/ddppo-models/${WEIGHTS_NAME} \
        TASK_CONFIG.DATASET.SCENES_DIR ${REPO_PATH}/data/scene_datasets \
        RL.DDPPO.backbone ${BACKBONE} \
        TASK_CONFIG.SEED ${SEED} \
        VIDEO_OPTION ${VIDEO_OPTION} \
        ${EXTRA_CMDS}"

    CMD_EVAL_OPTS="${CMD_COMMON_OPTS} \
        EVAL.SPLIT ${VAL_SPLIT} \
        TASK_CONFIG.DATASET.CONTENT_SCENES [\"*\"] \
        TEST_EPISODE_COUNT ${TEST_EPISODE_COUNT} \
        NUM_ENVIRONMENTS 1 \
        TASK_CONFIG.DATASET.DIFFICULTY ${DIFFICULTY} \
        RL.PPO.num_mini_batch 1 \
        TASK_CONFIG.DATASET.DATA_PATH ${REPO_PATH}/data/datasets/pointnav/${ENVIRONMENT}/v1/${VAL_SPLIT}/${VAL_SPLIT}.json.gz \
        TASK_CONFIG.TASK.TOP_DOWN_MAP.MAP_RESOLUTION 1024 \
        WANDB_NAME ${EXP_NAME}_${VAL_SPLIT}_${DIFFICULTY} \
        WANDB_MODE online \
        EVAL.USE_CKPT_CONFIG True"

    python -u -m run \
        --run-type eval ${CMD_EVAL_OPTS}

}
