import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from homography_utils import load, compute_homography, apply_homography

tgt_path = "target-2.png"
os.makedirs("points", exist_ok=True)
os.makedirs("src", exist_ok=True)
pts_path = os.path.join("points", tgt_path.replace(".png", "_pts.npz"))
fig_path = os.path.join("src", tgt_path.replace(".png", "_step1.png"))
saved =np.load(pts_path) if os.path.exists(pts_path) else None

src = load("source.png")
tgt = load(tgt_path)

def point_selection(img, n, title):
    plt.figure(figsize=(10, 7))
    plt.imshow(img)
    plt.title(title)
    pts = plt.ginput(n, timeout=0)
    plt.close()
    return np.array(pts)

src_pts = saved["src_pts"] if saved is not None else point_selection(src, 4, "Source Image Pts")
tgt_pts = saved["tgt_pts"] if saved is not None else point_selection(tgt, 4, "Target Image Pts")

H = compute_homography(src_pts, tgt_pts)

check_pts = saved["check_pts"] if saved is not None else point_selection(src, 6, "6 Test Points")
mapped = apply_homography(H, check_pts)

h, w = tgt.shape[:2]
warped = cv2.warpPerspective(src, H, (w, h))
overlay = cv2.addWeighted(tgt, 0.5, warped, 0.5, 0)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7))
ax1.imshow(tgt)
ax1.scatter(mapped[:, 0], mapped[:, 1], c="red", marker="x", s=100)
ax1.set_title("Source points mapped onto target")
ax2.imshow(overlay)
ax2.set_title("Warped source over target")
plt.tight_layout()
plt.savefig(fig_path)
plt.show()

H_cv, _ = cv2.findHomography(src_pts, tgt_pts, 0)
print(np.round(H, 4))
print(np.round(H_cv / H_cv[2, 2], 4))

np.savez(pts_path, src_pts=src_pts, tgt_pts=tgt_pts, check_pts=check_pts)

