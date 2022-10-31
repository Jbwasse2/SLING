import torch
from operator import itemgetter
import pickle
import seaborn as sns
import math
import matplotlib.pyplot as plt
import cv2
import numpy as np
import quaternion

from habitat.sims.habitat_simulator.actions import HabitatSimActions
from src.functions.validity_func.local_nav import LocalAgent, loop_nav
from src.functions.validity_func.validity_utils import (
    get_l2_distance,
    get_sim_location,
    get_rel_pose_change,
)
from src.utils.sim_utils import se3_to_mat, get_relative_location
from habitat_sim.geo import UP
from habitat_sim.utils.common import quat_from_angle_axis
from models.utils import (compute_pose_error, compute_epipolar_error,
                          estimate_pose, make_matching_plot,
                          error_colormap, AverageTimer, pose_auc, read_image,
                          rotate_intrinsics, rotate_pose_inplane,
                          scale_intrinsics, process_resize, frame2tensor)
from loftr.utils.plotting import make_matching_figure
from loftr.loftr import LoFTR, default_cfg
#from geoslam import get_slam_pose_labels

"""
predict if you want to switch to local navigation
"""


def ready_image(image, device, resize, rotation, resize_float):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype('float32'), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype('float32')

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    inp = frame2tensor(image, device)
    return image, inp, scales


def visualize_kp(image1, image2, kp1, kp2):
    for kp in kp1:
        image1[math.floor(kp[1]), math.floor(kp[0]), 0] = 255
        image1[math.floor(kp[1]), math.floor(kp[0]), 1] = 0
        image1[math.floor(kp[1]), math.floor(kp[0]), 2] = 0
    for kp in kp2:
        image2[math.floor(kp[1]), math.floor(kp[0]), 0] = 255
        image2[math.floor(kp[1]), math.floor(kp[0]), 1] = 0
        image2[math.floor(kp[1]), math.floor(kp[0]), 2] = 0
    im = np.hstack([image1, image2])
    for p1, p2 in zip(kp1, kp2):
        xs = [math.floor(p1[0]), 640 + math.floor(p2[0])]
        ys = [math.floor(p1[1]), math.floor(p2[1])]
        plt.plot(xs, ys, color='r')
    plt.imshow(im)
    plt.show()


def plot3d_in2d(pts, pts1hom, image):
    pts1hom = np.array(pts1hom)
    dist = pts1hom[:, 2]
    ret = [0 for _ in dist]
    a = sns.color_palette("flare", len(dist))
    keyList = sorted(enumerate(dist), key=itemgetter(1))
    for counter, (position, d) in enumerate(keyList):
        ret[position] = a[counter]
    ret_tup = []
    for r in ret:
        ret_tup.append((tuple([x for x in r])))
    fig, ax = plt.subplots()
    for kp, c in zip(pts, ret_tup):
        image[math.floor(kp[1]), math.floor(kp[0]), 0] = 255
        image[math.floor(kp[1]), math.floor(kp[0]), 1] = 0
        image[math.floor(kp[1]), math.floor(kp[0]), 2] = 0
        c = plt.Circle((kp[0], kp[1]), 5.0, color=c)
        ax.add_patch(c)
    plt.imshow(image)
    plt.show()


def plot_pcd(rgb1, d1, read_pinhole_camera_intrinsic):
    source_color = o3d.geometry.Image(rgb1)
    source_depth = o3d.geometry.Image(d1)
    rgbd_im1 = o3d.geometry.RGBDImage.create_from_color_and_depth(
        source_color, source_depth)
    cam = o3d.camera.PinholeCameraIntrinsic(
        o3d.camera.PinholeCameraIntrinsicParameters.PrimeSenseDefault)
    pcd1 = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_im1, cam)
    pcd1.transform([[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])
    o3d.io.write_point_cloud("test.pcd", pcd1)
    # o3d.visualization.draw_geometries([pcd1])


def get_target_cv2(img1, img2, d1, use_glue, use_loftr, device, matching, matcher, number_matches, no_switch):
    d1 = 10 * d1
    #K = np.array([[554.59, 0.0, 319.5], [0.0, 415.94, 239.5], [0.0, 0.0, 1.0]])
    K = np.array([[184.75, 0.0, 319.5], [0.0, 138.56, 239.5], [0.0, 0.0, 1.0]])
    # define constants
    mask = np.where(d1 == 0.0)
    d1[mask] = np.NaN

    if use_glue:
        # Get matches
        resize = [640, 480]
        img1, inp1, scales1 = ready_image(img1, device, resize, 0, False)
        img2, inp2, scales2 = ready_image(img2, device, resize, 0, False)
        pred = matching({'image0': inp1, 'image1': inp2})
        pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
        kp1, kp2 = pred['keypoints0'], pred['keypoints1']
        matches, match_confidence = pred['matches0'], pred['matching_scores0']
        pts1 = []
        pts2 = []
        pts1hom = []
        pts2hom = []
        for p1, conf, match in zip(kp1, match_confidence, matches):
            if conf >= 0.50:
                d = d1[int(p1[1]), int(p1[0])]
                if np.isnan(d):
                    continue
                Y = (p1[0] - K[0, 2]) * d / K[0, 0]
                X = -(p1[1] - K[1, 2]) * d / K[1, 1]
                Z = d
                pts1hom.append((X, Y, Z))
                pts1.append(p1)
                pts2.append(kp2[match])

    elif use_loftr:
        img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
        img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
        img0_raw = cv2.resize(img1, (640, 480))
        img1_raw = cv2.resize(img2, (640, 480))
        img0 = torch.from_numpy(img0_raw)[None][None].cuda() / 255.
        img1 = torch.from_numpy(img1_raw)[None][None].cuda() / 255.
        batch = {'image0': img0, 'image1': img1}
        with torch.no_grad():
            matcher(batch)
            kp1 = batch['mkpts0_f'].cpu().numpy()
            kp2 = batch['mkpts1_f'].cpu().numpy()
            match_confidence = batch['mconf'].cpu().numpy()
        pts1 = []
        pts2 = []
        pts1hom = []
        pts2hom = []
        for p1, conf, p2 in zip(kp1, match_confidence, kp2):
            if conf >= 0.0:
                d = d1[int(p1[1]), int(p1[0])]
                if np.isnan(d):
                    continue
                Y = (p1[0] - K[0, 2]) * d / K[0, 0]
                X = -(p1[1] - K[1, 2]) * d / K[1, 1]
                Z = d
                pts1hom.append((X, Y, Z))
                pts1.append(p1)
                pts2.append(p2)
    else:
        # Initiate SIFT detector
        sift = cv2.SIFT_create()

        # find the keypoints and descriptors with SIFT
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        if isinstance(des1, type(None)) or isinstance(des2, type(None)):
            #        print("Not enough features")
            return -1, -1, -1

        # find matches
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            matches = flann.knnMatch(des1, des2, k=2)
        except Exception as e:
            return -1, -1, -1

        # store all the good matches as per Lowe's ratio test.
        good = []
        pts1 = []
        pts2 = []
        pts1hom = []
        pts2hom = []
        # ratio test as per Lowe's paper
        for i, (m, n) in enumerate(matches):
            if m.distance < 0.75*n.distance:
                good.append([m])
                pt1 = kp1[m.queryIdx].pt
                pt2 = kp2[m.trainIdx].pt
                d = d1[int(pt1[1]), int(pt1[0])]
                if np.isnan(d):
                    continue
    #            X = (pt1[0] - K[0,2]) * d / K[0,0]
    #            Y = (pt1[1] - K[1,2]) * d / K[1,1]
                Y = (pt1[0] - K[0, 2]) * d / K[0, 0]
                X = -(pt1[1] - K[1, 2]) * d / K[1, 1]
                Z = d
                pts1hom.append((X, Y, Z))
                pts1.append(pt1)
                pts2.append(pt2)
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    r0 = np.array([0.0, 0.0, 0.0])
    t0 = np.array([0.0, 0.0, 0.0])

    pts1hom = np.array(pts1hom, dtype=np.float64).squeeze()
    if len(pts1hom) < number_matches and not no_switch:
        return -1, -1, -1
    dist_coeffs = np.zeros((4, 1))
    try:
        (success, R2, t2, mask) = cv2.solvePnPRansac(
            pts1hom, pts2, K, dist_coeffs, flags=1)
    except Exception as e:
        return -1, -1, -1
    if not success:
        return -1, -1, -1
    t2 = t2.flatten()
    #rho = np.linalg.norm(t2)
    t2 = np.array([t2[0], -t2[2]])
    rho = np.linalg.norm(t2)
    if rho >= 4.0 and not no_switch:
        return -1, -1, -1
    ref = np.array([0, 1])
    sign = np.sign(np.cross(t2, ref))
    phi = sign * np.arccos(np.dot(t2, ref) /
                           (np.linalg.norm(t2) * np.linalg.norm(ref)))
    M = cv2.Rodrigues(R2)[0]                                                                        
    sy = math.sqrt(M[0,0] * M[0,0] +  M[1,0] * M[1,0])                                              
    degRot = math.degrees(math.atan2(-M[2,0], sy))  
    #d1 = 0.1 * d1
    #plot_pcd(img1, d1, pinhole_camera_intrinsic)
    #plt.imshow(np.hstack([img1, img2]))
    # plt.show()
    mask = mask.flatten()
#    visualize_kp(img1, img2, pts1[mask], pts2[mask])
#    plot3d_in2d(pts1, pts1hom, img1)
    return rho, phi, degRot


def predict_end_exploration(args, agent, visualizer):
    switchThreshold = args.switch_threshold
    switch = False

    with torch.no_grad():
        batch_goal = agent.goal_feat.clone().detach()
        batch_nodes = (
            agent.node_feats[agent.current_node,
                             :].clone().detach().unsqueeze(0)
        )
        switch_pred, _, _ = agent.switch_model(
            batch_nodes.to(args.device), batch_goal.to(args.device)
        )
        switch_pred = switch_pred.detach().cpu()[0].item()
        gotoLocal = False
        if args.use_gt_sim:
            curr_position = agent.current_pos.numpy()
            curr_rotation = agent.current_rot.numpy()
            rho_gt, phi_gt = get_relative_location(
                curr_position, curr_rotation, agent.goal_pos)
            if rho_gt < 3.0:
                gotoLocal = True
        else:
            if switch_pred >= switchThreshold:
                gotoLocal = True
                

        if gotoLocal:
            switch = True
            agent.gt_count += 1
            agent.reached_goal_step = agent.steps
            rho, phi = agent.goal_model(
                batch_nodes.to(args.device), batch_goal.to(args.device)
            )
#            print(get_slam_rho_phi(agent))
            rho = rho.cpu().detach().item()
            phi = phi.cpu().detach().item()
            curr_position = agent.current_pos.numpy()
            curr_rotation = agent.current_rot.numpy()
            rho_gt, phi_gt = get_relative_location(
                curr_position, curr_rotation, agent.goal_pos)
            if args.nrns_switch:
                obs = agent.sim.get_observations_at(
                    curr_position, quaternion.from_float_array(curr_rotation))
                rho_, phi_, rot_ = get_target_cv2(
                    obs['rgb'], agent.goal_img, obs['depth'], True, False, agent.args.device, agent.super_glue, agent.matcher, args.number_of_matches,args.no_switch)
                if rho_ == -1 and phi_ == -1:
                    print("switchback1")
                    switch=False
#            f = open("rhophi_nrns.txt", "a")
#            f.write(str(rho) + "," + str(phi) + "," + str(rho_gt) + "," + str(phi_gt) + "\n")
#            f.close()
            if args.use_pnp_rhophi or args.use_glue_rhophi or args.use_loftr_rhophi:
                curr_position = agent.current_pos.numpy()
                curr_rotation = agent.current_rot.numpy()
                obs = agent.sim.get_observations_at(
                    curr_position, quaternion.from_float_array(curr_rotation))
                rot = 9001

                if args.use_rot:
                    ANGLE_THRESHOLD = math.radians(20)
                else:
                    ANGLE_THRESHOLD = 9000
                ROT_DIFF = math.radians(25)
                sim = agent.sim
                count = 0
                while abs(rot) > ANGLE_THRESHOLD:
                    rho, phi, rot = get_target_cv2(
                        obs['rgb'], agent.goal_img, obs['depth'], args.use_glue_rhophi, args.use_loftr_rhophi, agent.args.device, agent.super_glue, agent.matcher, args.number_of_matches,args.no_switch)
                    if rho == -1 and phi == -1:
                        if args.no_switch:
                            switch=True
                            rho = 0
                            phi = 0
                        else:
                            print("switchback2")
                            switch = False
                            return
                    break
                    rot_act = rot
                    while abs(rot_act) > ANGLE_THRESHOLD:
                        if rot_act > 0:
                            if agent.actuation_noise:
                                obs = sim.step(HabitatSimActions.TURN_RIGHT)
                            else:
                                obs = sim.step(3)
                            rot_act = rot_act - ROT_DIFF
                        else:
                            if agent.actuation_noise:
                                obs = sim.step(HabitatSimActions.TURN_LEFT)
                            else:
                                obs = sim.step(2)
                            rot_act = rot_act + ROT_DIFF
                    count += 1
                    if count > 15:
                        switch = False
                        return

                save_loc = "./GT_data/" + str(agent.scene_id)
                rho_gt, phi_gt = get_relative_location(
                    curr_position, curr_rotation, agent.goal_pos)
            if args.use_gt_rhophi:
                rho_guess = rho
                phi_guess = phi
                curr_position = agent.current_pos.numpy()
                curr_rotation = agent.current_rot.numpy()
                rho, phi = get_relative_location(
                    curr_position, curr_rotation, agent.goal_pos)
#            f = open("rhophi_pnp.txt", "a")
#            f.write(str(rho) + "," + str(phi) + "," + str(rho_gt) + "," + str(phi_gt) + "\n")
#            f.close()
            #DOUBLE CHECK
            foo = localnav(agent, rho, phi, visualizer)
#            if args.use_pnp_rhophi:
#                curr_position = agent.current_pos.numpy()
#                curr_rotation = agent.current_rot.numpy()
#                obs = agent.sim.get_observations_at(curr_position, quaternion.from_float_array(curr_rotation))
#                rho, phi = get_target_cv2(obs['rgb'], agent.goal_img, obs['depth'])
##                rho_gt, phi_gt = get_relative_location(curr_position, curr_rotation, agent.goal_pos)
##                print("guess " + str(rho) + " " + str(phi))
##                print("gt " + str(rho_gt) + " " + str(phi_gt))
##                print("gt disp " + str(agent.goal_pos - curr_position))
#                if rho == -1 and phi == -1:
#                    pass
#                else:
#                    foo = localnav(agent, rho, phi, visualizer)

        else:
            switch = False
    if switch and args.double_check_switch:
        new_batch_nodes = (
            agent.get_feat(agent.current_pos.numpy(),
                           agent.current_rot.numpy()).clone().detach()
        )
        batch_goal = agent.goal_feat.clone().detach()
        switch_pred, _, _ = agent.switch_model(
            new_batch_nodes.to(args.device), batch_goal.to(args.device)
        )
        switch_pred = switch_pred.detach().cpu()[0].item()
        if switch_pred < switchThreshold:
            switch = False

    return switch


def get_slam_rho_phi(agent):
    # def get_slam_pose_labels(env, agent, sim, rgb=None, depth=None):
    curr_position = agent.current_pos.numpy()
    curr_rotation = agent.current_rot.numpy()
    obs = agent.sim.get_observations_at(
        curr_position, quaternion.from_float_array(curr_rotation))
    goal_position = agent.goal_pos
    goal_rotation = agent.goal_rot
    goal_obs = agent.sim.get_observations_at(
        goal_position, quaternion.from_float_array(goal_rotation))
    obs["rgb"] = cv2.resize(obs["rgb"], dsize=(256, 256))
    obs["depth"] = cv2.resize(obs["depth"], dsize=(256, 256))
    goal_obs["rgb"] = cv2.resize(goal_obs["rgb"], dsize=(256, 256))
    goal_obs["depth"] = cv2.resize(goal_obs["depth"], dsize=(256, 256))
    rgb = [obs['rgb'], goal_obs['rgb']]
    depth = [obs['depth'], goal_obs['depth']]
    pose6D = get_slam_pose_labels(
        agent.scan_name, 'orbslam2-rgbd', agent.sim, rgb, depth)
    return pose6D


def local_goal_pose(phi, rho, start_pos, start_rot):
    stateA = se3_to_mat(
        quaternion.from_float_array(start_rot),
        np.asarray(start_pos),
    )
    stateB = (
        stateA
        @ se3_to_mat(
            quat_from_angle_axis(phi, UP),
            np.asarray([0, 0, 0]),
        )
        @ se3_to_mat(
            quaternion.from_float_array([1, 0, 0, 0]),
            np.asarray([0, 0, -1 * rho]),
        )
    )
    final_pos = stateB[0:3, 3]
    return final_pos


def localnavDDPPO(agent, rho, phi, visualizer):
    prev_poses = []
    nav_length = 0.0
    sim = agent.sim
    curr_position = agent.current_pos.numpy()
    curr_rotation = agent.current_rot.numpy()
    previous_pose = curr_position
    not_done_masks = torch.zeros(1, 1).to(agent.args.device)
    not_done_masks += 1
    prev_action = torch.zeros(1, 1).to(agent.args.device)
    obs = sim.get_observations_at(
        curr_position, quaternion.from_float_array(curr_rotation))
    displacement = np.array([rho, phi])
    displacement = torch.from_numpy(displacement).type(torch.float32)
    goal_pose = local_goal_pose(phi, rho, curr_position, curr_rotation)
    # Can use curr_rotation because doesn't matter what end rotation is...
    goal_pose = get_sim_location(
        goal_pose, quaternion.from_float_array(curr_rotation)
    )
    obs["pointgoal_with_gps_compass"] = displacement.unsqueeze(
        0).to(agent.args.device)
    obs["depth"] = cv2.resize(obs["depth"], dsize=(256, 256))
    obs["rgb"] = cv2.resize(obs["rgb"], dsize=(256, 256))
    obs["depth"] = torch.from_numpy(obs["depth"]).unsqueeze(
        0).unsqueeze(-1).to(agent.args.device)
    for i in range(min(100, 499 - agent.steps)):
        with torch.no_grad():
            _, action, _, agent.ddppo_hidden = agent.ddppo.act(
                obs, agent.ddppo_hidden, prev_action, not_done_masks, deterministic=True
            )
            action = action[0].item()
            prev_poses.append([curr_position, curr_rotation])
            if agent.actuation_noise:
                if action == 1:
                    obs = sim.step(HabitatSimActions.MOVE_FORWARD)
                elif action == 2:
                    obs = sim.step(HabitatSimActions.TURN_LEFT)
                elif action == 3:
                    obs = sim.step(HabitatSimActions.TURN_RIGHT)
            else:
                # For some reason stop doesn't work? But we take care of it later
                if action is not 0:
                    obs = sim.step(action)
            prev_poses.append([curr_position, curr_rotation])
            prev_pos = curr_position
            curr_position = sim.get_agent_state().position
            curr_rotation = quaternion.as_float_array(
                sim.get_agent_state().rotation)
            if agent.pose_noise:
                curr_position = np.asarray(
                    agent.noisy_sensor.get_noisy_pose(
                        action, previous_pose, curr_position
                    )
                )
            updated_pose = get_sim_location(
                curr_position, quaternion.from_float_array(curr_rotation)
            )
            nav_length += np.linalg.norm(prev_pos - curr_position)
            prev_action[0] = action
            curr_position = agent.current_pos.numpy()
            curr_rotation = agent.current_rot.numpy()
            rho, phi = get_relative_location(
                curr_position, curr_rotation, agent.goal_pos)
            displacement = np.array([rho, phi])
            displacement = torch.from_numpy(displacement).type(torch.float32)
            DISTANCE = 0.8
            if displacement[0] <= DISTANCE or action == 0:
                break
            obs["pointgoal_with_gps_compass"] = displacement.unsqueeze(
                0).to(agent.args.device)
            obs["depth"] = cv2.resize(obs["depth"], dsize=(256, 256))
            obs["rgb"] = cv2.resize(obs["rgb"], dsize=(256, 256))
            obs["depth"] = torch.from_numpy(obs["depth"]).unsqueeze(
                0).unsqueeze(-1).to(agent.args.device)
    if agent.visualize:
        run_vis(agent, visualizer, prev_poses)
        agent.prev_poses.extend(prev_poses)
    agent.current_pos = torch.tensor(curr_position)
    agent.current_rot = torch.tensor(curr_rotation)
    agent.length_taken += nav_length


def run_vis(agent, visualizer, prev_poses):
    for p in prev_poses:
        img = agent.sim.get_observations_at(p[0], quaternion.from_float_array(p[1]),)[
            "rgb"
        ][:, :, :3]
        agent.current_pos, agent.current_rot = torch.tensor(
            p[0]), torch.tensor(p[1])
        visualizer.seen_images.append(img)
        visualizer.current_graph(agent, switch=True)


def localnav(agent, rho, phi, visualizer):
    agent.sim.set_agent_state(
        agent.current_pos.numpy(),
        quaternion.from_float_array(agent.current_rot.numpy()),
    )
    agent.switch_index = len(agent.prev_poses)
    agent.prev_poses.append(
        [agent.current_pos.numpy(), agent.current_rot.numpy()])
    agent.explore_exploit.append(1)
    try:
        agent.sim.set_agent_state(
            agent.current_pos.numpy(),
            quaternion.from_float_array(agent.current_rot.numpy()),
        )
        local_agent = LocalAgent(
            agent.actuation_noise,
            agent.pose_noise,
            agent.current_pos.numpy(),
            agent.current_rot.numpy(),
            map_size_cm=1200,
            map_resolution=5,
        )
        final_pos, final_rot, nav_length, prev_poses = loop_nav(
            agent.sim,
            local_agent,
            agent.current_pos.numpy(),
            agent.current_rot.numpy(),
            rho,
            phi,
            min(100, 499 - agent.steps),
        )
        agent.explore_exploit.extend([1 for _ in range(len(prev_poses))])
        agent.steps += local_agent.steps
        if agent.visualize:
            run_vis(agent, visualizer, prev_poses)
        agent.prev_poses.extend(prev_poses)
        agent.current_pos = torch.tensor(final_pos)
        agent.current_rot = torch.tensor(final_rot)
        agent.length_taken += nav_length
        return np.linalg.norm(agent.goal_pos - final_pos)
    except:
        print("ERROR: local navigation through error")

    return np.linalg.norm(agent.goal_pos - agent.current_pos.numpy())
