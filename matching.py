import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from homography_utils import load, compute_homography, apply_homography

TARGETS = ["target-1.png", "target-2.png", "target-3.png"]
DETECTORS = {
    "sift": (cv2.SIFT_create(), cv2.NORM_L2),
    "orb": (cv2.ORB_create(5000), cv2.NORM_HAMMING),
}
RATIO = 0.75
INLIER_PX = 3.0
ITERS = 2000
FIG_DIR = os.path.join("src", "step2")

def match_features(detector, norm, img1, img2):
    gray1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)
    kp1, des1 = detector.detectAndCompute(gray1, None)
    kp2, des2 = detector.detectAndCompute(gray2, None)
    pairs = cv2.BFMatcher(norm).knnMatch(des1, des2, k=2)
    good = [p[0] for p in pairs if len(p) == 2 and p[0].distance < RATIO * p[1].distance]
    return kp1, kp2, good

def ransac_homography(src_pts, dst_pts, rng):
    best = np.zeros(len(src_pts), dtype=bool)
    for _ in range(ITERS):
        idx = rng.choice(len(src_pts), 4, replace=False)
        H = compute_homography(src_pts[idx], dst_pts[idx])
        with np.errstate(divide="ignore", invalid="ignore"):
            errs = np.linalg.norm(apply_homography(H, src_pts) - dst_pts, axis=1)
        inliers = errs < INLIER_PX
        if inliers.sum() > best.sum():
            best = inliers
    return compute_homography(src_pts[best], dst_pts[best]), best

def mean_diff(H1, H2, pts):
    return np.linalg.norm(apply_homography(H1, pts) - apply_homography(H2, pts), axis=1).mean()

def save_figure(src, tgt, kp1, kp2, good, inliers, H, name):
    inlier_matches = [m for m, keep in zip(good, inliers) if keep]
    drawn = cv2.drawMatches(src, kp1, tgt, kp2, inlier_matches, None,
                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    h, w = tgt.shape[:2]
    warped = cv2.warpPerspective(src, H, (w, h))
    overlay = cv2.addWeighted(tgt, 0.5, warped, 0.5, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), width_ratios=[2, 1])
    ax1.imshow(drawn)
    ax1.set_title(f"{len(inlier_matches)} inlier matches")
    ax2.imshow(overlay)
    ax2.set_title("Warped source over target")
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, name))
    plt.close(fig)

os.makedirs(FIG_DIR, exist_ok=True)
rng = np.random.default_rng(0)
src = load("source.png")
h, w = src.shape[:2]
corners = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=float)

print(f"{'target':<14}{'feat':<6}{'kp src':>8}{'kp tgt':>8}{'matches':>9}{'inliers':>9}{'vs click':>10}{'vs cv2':>9}")
for tgt_path in TARGETS:
    tgt = load(tgt_path)
    pts_path = os.path.join("points", tgt_path.replace(".png", "_pts.npz"))
    saved = np.load(pts_path) if os.path.exists(pts_path) else None

    for feat, (detector, norm) in DETECTORS.items():
        kp1, kp2, good = match_features(detector, norm, src, tgt)
        row = f"{tgt_path:<14}{feat:<6}{len(kp1):>8}{len(kp2):>8}{len(good):>9}"
        if len(good) < 4:
            print(row + "  too few matches")
            continue

        src_pts = np.float64([kp1[m.queryIdx].pt for m in good])
        tgt_pts = np.float64([kp2[m.trainIdx].pt for m in good])
        H, inliers = ransac_homography(src_pts, tgt_pts, rng)

        H_click = compute_homography(saved["src_pts"], saved["tgt_pts"]) if saved is not None else None
        vs_click = f"{mean_diff(H, H_click, saved['check_pts']):.2f}" if saved is not None else "n/a"
        H_cv, _ = cv2.findHomography(src_pts, tgt_pts, cv2.RANSAC, INLIER_PX)
        vs_cv = f"{mean_diff(H, H_cv, corners):.2f}" if H_cv is not None else "n/a"
        print(row + f"{inliers.sum():>9}{vs_click:>10}{vs_cv:>9}")

        name = tgt_path.replace(".png", f"_{feat}_step2.png")
        save_figure(src, tgt, kp1, kp2, good, inliers, H, name)
