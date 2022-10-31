import numpy as np
from src.utils.sim_utils import set_up_habitat
import pudb

sim_dir = "../../data/scene_datasets/gibson_train_val"
house = "Cantwell"
scene = "{}/{}.glb".format(sim_dir, house)
sim, _ = set_up_habitat(scene)
geo_dist = sim.geodesic_distance(pointA, pointB)
