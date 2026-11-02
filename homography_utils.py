import numpy as np
import cv2

def load(path):
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

def compute_homography(src_pts, dst_pts):
    A = []
    for (x, y), (xp, yp) in zip(src_pts, dst_pts):
        A.append([-x, -y, -1,  0,  0,  0, xp*x, xp*y, xp])
        A.append([ 0,  0,  0, -x, -y, -1, yp*x, yp*y, yp])
    _, _, Vt = np.linalg.svd(np.array(A))
    H = Vt[-1].reshape(3, 3)
    return H / H[2, 2]

def apply_homography(H, pts):
    pts_h = np.hstack([pts, np.ones((len(pts), 1))])
    out = (H @ pts_h.T).T
    return out[:, :2] / out[:, 2:3]
