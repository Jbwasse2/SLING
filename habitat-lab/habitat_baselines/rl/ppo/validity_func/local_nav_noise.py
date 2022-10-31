import quaternion
import matplotlib.pyplot as plt
import pickle
import numpy as np
import cv2
import skimage
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat_baselines.rl.ppo.validity_func.fmm_plannermine import FMMPlanner
from src.functions.validity_func.map_builder import build_mapper
from src.functions.validity_func.validity_utils import (
    get_l2_distance,
    get_sim_location,
    get_rel_pose_change,
)
class NoisySensor:
    def __init__(self, noise_level):
        self.noise_level = noise_level
        self.noise_dir = "./models/noise_models/"
        self.sensor_noise_fwd = pickle.load(
            open(self.noise_dir + "sensor_noise_fwd.pkl", "rb")
        )
        self.sensor_noise_right = pickle.load(
            open(self.noise_dir + "sensor_noise_right.pkl", "rb")
        )
        self.sensor_noise_left = pickle.load(
            open(self.noise_dir + "sensor_noise_left.pkl", "rb")
        )

    def get_l2_distance(self, x1, x2, y1, y2):
        """
        Computes the L2 distance between two points.
        """
        return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** 0.5

    def get_rel_pose_change(self, pos2, pos1):
        x1, y1, o1 = pos1
        x2, y2, o2 = pos2
        theta = np.arctan2(y2 - y1, x2 - x1) - o1
        dist = self.get_l2_distance(x1, x2, y1, y2)
        dx = dist * np.cos(theta)
        dy = dist * np.sin(theta)
        do = o2 - o1
        return dx, dy, do

    def get_noisy_sensor_readings(self, action, gt_pose_change):
        dx_gt, dy_gt, do_gt = gt_pose_change
        if action == 1:  ## Forward
            x_err, y_err, o_err = self.sensor_noise_fwd.sample()[0][0]
        elif action == 2:  ## Left
            x_err, y_err, o_err = self.sensor_noise_left.sample()[0][0]
        elif action == 3:  ## Right
            x_err, y_err, o_err = self.sensor_noise_right.sample()[0][0]
        else:  ##Stop
            x_err, y_err, o_err = 0.0, 0.0, 0.0

        x_err = x_err * self.noise_level
        y_err = y_err * self.noise_level
        o_err = o_err * self.noise_level
        return dx_gt + x_err, dy_gt + y_err, do_gt + np.deg2rad(o_err)

    def get_new_pose(self, pose, rel_pose_change):
        x, y, o = pose
        dx, dy, do = rel_pose_change
        global_dx = dx * np.sin(o) + dy * np.cos(o)
        global_dy = dx * np.cos(o) - dy * np.sin(o)
        x += global_dy
        y += global_dx
        o += do
        return x, y, o

    def get_noisy_pose(self, action, previous_pose, pose):
        gt_pose_change = self.get_rel_pose_change(pose, previous_pose)
        noisy_pose_change = self.get_noisy_sensor_readings(action, gt_pose_change)
        noisy_pose = np.asarray(self.get_new_pose(previous_pose, noisy_pose_change))
        # import ipdb

        # ipdb.set_trace()
        return noisy_pose


class LocalAgent(object):
    def __init__(
        self,
        curr_pos,
        curr_rot,
        map_size_cm,
        map_resolution,
    ):
        self.noisy_sensor = NoisySensor(noise_level=1.0)
        self.mapper = build_mapper()
        self.curr_pos = curr_pos
        self.curr_rot = curr_rot
        self.map_size_cm = map_size_cm
        self.map_resolution = map_resolution
        self.sim_origin = get_sim_location(
            self.curr_pos, quaternion.from_float_array(self.curr_rot)
        )
        self.collision = False
        self.initialize_local_map_pose()
        self.stg_x, self.stg_y = int(self.y_gt / map_resolution), int(
            self.x_gt / map_resolution
        )
        self.new_sim_origin = self.sim_origin
        self.reset_goal = True
        self.steps = 0

    def initialize_local_map_pose(self):
        self.mapper.reset_map()
        self.x_gt, self.y_gt, self.o_gt = (
            self.map_size_cm / 100.0 / 2.0,
            self.map_size_cm / 100.0 / 2.0,
            0.0,
        )
        self.reset_goal = True
        self.sim_origin = get_sim_location(
            self.curr_pos, quaternion.from_float_array(self.curr_rot)
        )
        self.local_locs = np.zeros(self.mapper.get_map()[:, :, 1].shape)

    def update_local_map(self, curr_depth_img):
        self.x_gt, self.y_gt, self.o_gt = self.get_mapper_pose_from_sim_pose(
            self.new_sim_origin,
            self.sim_origin,
        )

        x, y, o = self.x_gt, self.y_gt, self.o_gt
        _, self.local_map, _, self.local_exp_map, _ = self.mapper.update_map(
            curr_depth_img[:, :, 0] * 1000.0, (x, y, o)
        )

        if self.collision:
            self.mapper.map[self.stg_x, self.stg_y, 1] = 10.0
            self.collision = False

    def get_mapper_pose_from_sim_pose(self, sim_pose, sim_origin):
        x, y, o = get_rel_pose_change(sim_pose, sim_origin)
        return (
            self.map_size_cm - (x * 100.0 + self.map_size_cm / 2.0),
            self.map_size_cm - (y * 100.0 + self.map_size_cm / 2.0),
            o,
        )

    def set_goal(self, delta_dist, delta_rot):
        start = (
            int(self.y_gt / self.map_resolution),
            int(self.x_gt / self.map_resolution),
        )
        goal = (
            start[0]
            + int(
                delta_dist
                * np.sin(delta_rot + self.o_gt)
                * 100.0
                / self.map_resolution
            ),
            start[1]
            + int(
                delta_dist
                * np.cos(delta_rot + self.o_gt)
                * 100.0
                / self.map_resolution
            ),
        )
        self.goal = goal

    def navigate_local(self):
        traversible = (
            skimage.morphology.binary_dilation(
                self.local_map, skimage.morphology.disk(2)
            )
            != True
        )

        start = (
            int(self.y_gt / self.map_resolution),
            int(self.x_gt / self.map_resolution),
        )
        try:
            traversible[
                start[0] - 2 : start[0] + 3, start[1] - 2 : start[1] + 3
            ] = 1
        except:
            import ipdb

            ipdb.set_trace()
        planner = FMMPlanner(traversible, 360 // 10, 1)

        if self.reset_goal:
            planner.set_goal(self.goal, auto_improve=True)
            self.goal = planner.get_goal()
            self.reset_goal = False
        else:
            planner.set_goal(self.goal, auto_improve=True)

        stg_x, stg_y = start
        stg_x, stg_y, replan = planner.get_short_term_goal2((stg_x, stg_y))

        if get_l2_distance(start[0], self.goal[0], start[1], self.goal[1]) < 3:
            terminate = 1
        else:
            terminate = 0

        agent_orientation = np.rad2deg(self.o_gt)
        action = planner.get_next_action(
            start, (stg_x, stg_y), agent_orientation
        )
        self.stg_x, self.stg_y = int(stg_x), int(stg_y)
        return action, terminate

    def get_map(self):
        self.stg_x, self.stg_y = int(self.y_gt / self.map_resolution), int(
            self.x_gt / self.map_resolution
        )
        metric_map = self.local_map + 0.5 * self.local_exp_map * 255
        metric_map[
            int(self.stg_x) - 1 : int(self.stg_x) + 1,
            int(self.stg_y) - 1 : int(self.stg_y) + 1,
        ] = 255
        metric_map = cv2.resize(metric_map / 255.0, (80, 80))

        # metric_map = self.local_map + 0.5 * self.local_exp_map * 255
        # metric_map = metric_map.astype("uint8")
        # metric_map = cv2.cvtColor(metric_map, cv2.COLOR_GRAY2RGB)
        # self.stg_x, self.stg_y = int(self.y_gt / self.map_resolution), int(
        #     self.x_gt / self.map_resolution
        # )
        # metric_map[
        #     int(self.stg_x) - 1 : int(self.stg_x) + 1,
        #     int(self.stg_y) - 1 : int(self.stg_y) + 1,
        #     :,
        # ] = [
        #     0,
        #     0,
        #     255,
        # ]
        # metric_map = cv2.resize(metric_map, (80, 80))
        # metric_map = metric_map.astype("float")
        # metric_map /= 255
        ##
        # metric_map = torch.from_numpy(metric_map)
        return metric_map


def loop_nav(
    env,
    local_agent,
    start_pos,
    start_rot,
    delta_dist,
    delta_rot,
    max_steps,
    observation,
):
    prev_poses = []
    images = []
    curr_pos = observation['gps']
    curr_rot = observation['compass']
    nav_length = 0.0
    terminate_local = 0
    curr_depth_img = observation["depth"]
    local_agent.update_local_map(curr_depth_img)
    local_agent.set_goal(delta_dist, delta_rot)
    action, terminate_local = local_agent.navigate_local()
    previous_pose = curr_pos
    count = 0
    for _ in range(max_steps):
        observation = env.step([action])
        images.append(observation[0][0]['rgb'])
        curr_depth_img = observation[0][0]["depth"]
        curr_pos = observation[0][0]['gps']
        curr_rot = observation[0][0]['compass']
        curr_pos = self.noisy_sensor.get_noisy_pose(action, previous_pose, curr_pose)
        local_agent.new_sim_origin = get_sim_location(curr_pos, quaternion.from_float_array(curr_rot))
        previous_pose = curr_pos
        local_agent.update_local_map(curr_depth_img)
        action, terminate_local = local_agent.navigate_local()
        count += 1
        if terminate_local == 1:
            break

    return observation, images


def map_from_actions(sim, local_agent, start_pos, start_rot, action_list):
    maps = []
    obs = sim.get_observations_at(
        start_pos, quaternion.from_float_array(start_rot)
    )
    curr_depth_img = obs["depth"]
    local_agent.update_local_map(curr_depth_img)
    maps.append(local_agent.get_map())
    for action in action_list:
        obs = sim.step(action)
        curr_depth_img = obs["depth"]
        curr_position = sim.get_agent_state().position
        curr_rotation = sim.get_agent_state().rotation
        local_agent.new_sim_origin = get_sim_location(
            curr_position, curr_rotation
        )
        local_agent.update_local_map(curr_depth_img)
        maps.append(local_agent.get_map())
    return maps
