#!/usr/bin/env python

import cv2
import numpy as np
import torch


def process_resize(w, h, resize):
    assert len(resize) > 0 and len(resize) <= 2
    if len(resize) == 1 and resize[0] > -1:
        scale = resize[0] / max(h, w)
        w_new, h_new = int(round(w * scale)), int(round(h * scale))
    elif len(resize) == 1 and resize[0] == -1:
        w_new, h_new = w, h
    else:  # len(resize) == 2:
        w_new, h_new = resize[0], resize[1]

    # Issue warning if resolution is too small or too large.
    if max(w_new, h_new) < 160:
        print("Warning: input resolution is very small, results may vary")
    elif max(w_new, h_new) > 2000:
        print("Warning: input resolution is very large, results may vary")

    return w_new, h_new


def frame2tensor(frame, device):
    return torch.from_numpy(frame / 255.0).float()[None, None].to(device)


def ready_image(image, device, resize, rotation, resize_float):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    if image is None:
        return None, None, None
    w, h = image.shape[1], image.shape[0]
    w_new, h_new = process_resize(w, h, resize)
    scales = (float(w) / float(w_new), float(h) / float(h_new))

    if resize_float:
        image = cv2.resize(image.astype("float32"), (w_new, h_new))
    else:
        image = cv2.resize(image, (w_new, h_new)).astype("float32")

    if rotation != 0:
        image = np.rot90(image, k=rotation)
        if rotation % 2:
            scales = scales[::-1]
    inp = frame2tensor(image, device)
    return image, inp, scales


def get_target(img1, img2, d1, K, device, superglue):
    # Get matches
    resize = [int(img1.shape[1]), int(img1.shape[0])]
    _, inp1, scales1 = ready_image(img1, device, resize, 0, False)
    _, inp2, scales2 = ready_image(img2, device, resize, 0, False)
    nn_input = {"image0": inp1, "image1": inp2}
    pred = superglue(nn_input)
    pred = {k: v[0].cpu().detach().numpy() for k, v in pred.items()}
    kp1, kp2 = pred["keypoints0"], pred["keypoints1"]
    matches, match_confidence = (
        pred["matches0"],
        pred["matching_scores0"],
    )
    pts1 = []
    pts2 = []
    pts1hom = []
    for pt1, conf, match in zip(kp1, match_confidence, matches):
        if conf >= 0.50:
            pt2 = kp2[match]
            d = d1[int(pt1[1]), int(pt1[0])]
            if np.isnan(d):
                continue
            X = (pt1[0] - K[0, 2]) * d / K[0, 0]
            Y = (pt1[1] - K[1, 2]) * d / K[1, 1]
            Z = d
            p3d_1 = (X.item(), Y.item(), Z.item())
            pts1hom.append(p3d_1)
            pts1.append(pt1)
            pts2.append(pt2)
    pts1 = np.float64(pts1)
    pts2 = np.float64(pts2)
    pts1hom = np.array(pts1hom, dtype=np.float64).squeeze()
    dist_coeffs = np.zeros((4, 1))
    #print("CHANGE 00 to 50")
    if len(pts1hom) < 20:
        return -1
    try:
        (success, R2, t2, mask) = cv2.solvePnPRansac(
            pts1hom, pts2, K, dist_coeffs, flags=1
        )
    except Exception as e:
        return -1
    if not success:
        return -1
    t2 = t2.flatten()
    t2 = np.array([t2[0], -t2[2]])
    rho = np.linalg.norm(t2)
    if rho >= 4.0:
        return -1
    ref = np.array([0, 1])
    sign = np.sign(np.cross(t2, ref))
    phi = sign * np.arccos(
        np.dot(t2, ref) / (np.linalg.norm(t2) * np.linalg.norm(ref))
    )
    return np.array([rho, phi])
