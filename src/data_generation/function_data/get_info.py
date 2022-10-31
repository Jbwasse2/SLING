import numpy as np
import matplotlib.pyplot as plt
import re
import torch
from tqdm import tqdm
from pathlib import Path
from glob import glob
import gzip
import json
import msgpack_numpy
import pudb
import pdb
from src.utils.model_utils import load_places_resnet, get_res_feats
from src.utils.sim_utils import set_up_habitat
DEBUG = False
RESNET = load_places_resnet()


def get_sim(house):
    sim_dir = "../../data/scene_datasets/gibson_train_val"
    scene = "{}/{}.glb".format(sim_dir, house)
    sim, _ = set_up_habitat(scene)
    return sim

def check_match_resnet(states, RESNET, given_feats, sim):
    filePath = "./foo.pt"
    counter = 0
    wrong = []
    sumall1 = []
    sumall2 = []
    for state, feat_goal in zip(states, given_feats):
        sim.reset()
        p, r = state
        obs = sim.get_observations_at(p, r)['rgb']
        feats = get_res_feats( obs, RESNET).squeeze(0)
        sumall1.append(torch.sum(feats).item())
        sumall2.append(torch.sum(feat_goal).item())
        if not torch.allclose(feats, feat_goal):
            wrong.append(counter)
        counter += 1


def get_info(filename, save_dir):
    with gzip.open(filename, "r") as fin:
        episodes = json.loads(fin.read().decode("utf-8"))
    for count, e in tqdm(enumerate(episodes)):
        r = '^(.*)_'
        name = re.findall(r, e['episode_id'])[0]
        if DEBUG:
            sim = get_sim(name)
        flag = 0
        info = {}
        info['states'] = []
        info['actions'] = []
        g_t = e['goals'][0][0]
        g_r = e['goals'][0][1]
        for count2, (p, r) in enumerate(zip(e['poses'], e['rotations'])):
            new = [ p, r ]
            info['states'].append(new)
            if p == g_t and g_r == r:
                if flag == 0 and (count2+1 != len(e['poses']) or len(e['goals']) == 2):
                    new = [ p, r ]
                    info['states'].append(new)
                flag += 1
#                if flag == 1 and len(e['goals']) == 1 and count2 + 1 == len(e['poses']):
#                    new = [ p, r ]
#                    info['states'].append(new)

        
        info['actions'].extend(e['actions'])
        name = Path(Path(filename).stem).stem
        outfile = save_dir + name + "_" + str(count) + ".msg"
        scene = str(name) + "_" + str(count)
        trajectory_data_dir = "../../data/bar/foo/gibson/trajectory_data/"
        featFile = trajectory_data_dir + "trajectoryFeats/" + scene + ".pt"
        feats = torch.load(featFile).squeeze(-1).squeeze(-1)
        if DEBUG:
            check_match_resnet(info['states'], RESNET, feats, sim)
        if len(info['states']) == feats.shape[0] - 1:
            new = info['states'][-1]
            info['states'].append(new)
        if (flag == 1 and len(e['goals']) == 1) or (flag == 2 and len(e['goals']) == 2) or (flag == 1 and len(e['goals']) == 2 and count2+1 == len(e['poses'])):
            if feats.shape[0] == len(info['states']):
                msgpack_numpy.pack(info, open(outfile, "wb"), use_bin_type=True)
            else:
                pu.db
        else:
            pu.db
        if DEBUG:
            sim.close()

files = glob("../../data/bar/foo/gibson/trajectory_data/train_instances/*.json.gz")
save_dir = "../../data/bar/foo/gibson/trajectory_data/trajectoryInfo/"
#get_info("../../data/bar/foo/gibson/trajectory_data/train_instances/Cantwell.json.gz", None)
for f in tqdm(files):
    get_info(f, save_dir)

