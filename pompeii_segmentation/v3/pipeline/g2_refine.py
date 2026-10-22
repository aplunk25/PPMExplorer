"""Refine georeference by matching P-LOD room polygons to segmented rooms (IoU), then a robust
2nd-order polynomial fit pixel -> metric. Iterates matching <-> fitting."""
import json, numpy as np, cv2
from gcommon import *
closed = np.load('closed.npy')
H, W = closed.shape
free = (1 - closed).astype(np.uint8)
n, lab, st, cen = cv2.connectedComponentsWithStats(free, connectivity=4, ltype=cv2.CV_32S)
del free
area = st[:, 4]
plod = load_plod(('room',))
print('plod room rings', len(plod))

SC = 10000.0
def feats(c, r):
    c = np.asarray(c, float) / SC; r = np.asarray(r, float) / SC
    return np.stack([np.ones_like(c), c, r, c * c, c * r, r * r], 1)
def ifeats(x, y):
    x = np.asarray(x, float) / 1000; y = np.asarray(y, float) / 1000
    return np.stack([np.ones_like(x), x, y, x * x, x * y, y * y], 1)

A = coarse_affine()
def fwd_aff(c, r): return A[0, 0] * c + A[0, 1] * r + A[0, 2], A[1, 0] * c + A[1, 1] * r + A[1, 2]
Ai = np.linalg.inv(np.vstack([A, [0, 0, 1]]))[:2]
def inv_aff(x, y): return Ai[0, 0] * x + Ai[0, 1] * y + Ai[0, 2], Ai[1, 0] * x + Ai[1, 1] * y + Ai[1, 2]
fwd, inv = fwd_aff, inv_aff

for it in range(5):
    P_pix, P_m, ious, ids = [], [], [], []
    for props, xy in plod:
        c, r = inv(xy[:, 0], xy[:, 1])
        pts = np.stack([c, r], 1)
        x0, y0 = np.floor(pts.min(0)).astype(int) - 2; x1, y1 = np.ceil(pts.max(0)).astype(int) + 2
        if x0 < 0 or y0 < 0 or x1 >= W or y1 >= H: continue
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        cv2.fillPoly(m, [np.round((pts - [x0, y0]) * 4).astype(np.int32)], 1, shift=2)
        sub = lab[y0:y1, x0:x1][m == 1]
        if len(sub) < 50: continue
        bc = np.bincount(sub[sub > 0])
        if len(bc) == 0: continue
        k = int(bc.argmax()); inter = bc[k]
        if area[k] > 400000: continue
        # IoU against free-space-only part of the plod polygon (plod polygons include walls)
        pfree = int((sub > 0).sum())
        iou = inter / (pfree + area[k] - inter)
        if iou < 0.55: continue
        # centroid of our room (pixel) <-> centroid of plod ring (metric, area-weighted)
        xs, ys = xy[:, 0], xy[:, 1]
        a2 = xs[:-1] * ys[1:] - xs[1:] * ys[:-1]
        if abs(a2.sum()) < 1e-6: continue
        cxm = ((xs[:-1] + xs[1:]) * a2).sum() / (3 * a2.sum()); cym = ((ys[:-1] + ys[1:]) * a2).sum() / (3 * a2.sum())
        P_pix.append(cen[k]); P_m.append((cxm, cym)); ious.append(iou); ids.append((props['plod_id'], k))
    P_pix = np.array(P_pix); P_m = np.array(P_m); ious = np.array(ious)
    # robust fit (iteratively drop > 3*MAD residuals)
    keep = np.ones(len(P_pix), bool)
    for _ in range(6):
        X = feats(P_pix[keep, 0], P_pix[keep, 1])
        cx, *_ = np.linalg.lstsq(X, P_m[keep, 0], rcond=None); cy, *_ = np.linalg.lstsq(X, P_m[keep, 1], rcond=None)
        Xa = feats(P_pix[:, 0], P_pix[:, 1])
        res = np.hypot(Xa @ cx - P_m[:, 0], Xa @ cy - P_m[:, 1])
        thr = max(1.0, 3 * 1.4826 * np.median(np.abs(res[keep] - np.median(res[keep]))) + np.median(res[keep]))
        keep = res < thr
    Xi = ifeats(P_m[keep, 0], P_m[keep, 1])
    ic, *_ = np.linalg.lstsq(Xi, P_pix[keep, 0], rcond=None); ir, *_ = np.linalg.lstsq(Xi, P_pix[keep, 1], rcond=None)
    fwd = lambda c, r, cx=cx, cy=cy: (feats(c, r) @ cx, feats(c, r) @ cy)
    inv = lambda x, y, ic=ic, ir=ir: (ifeats(x, y) @ ic, ifeats(x, y) @ ir)
    print(f'iter {it}: matched {len(P_pix)}  inliers {keep.sum()}  median IoU {np.median(ious):.2f}  '
          f'residual median {np.median(res[keep]):.2f} m  RMS {np.sqrt((res[keep]**2).mean()):.2f} m  p95 {np.percentile(res[keep],95):.2f} m')

json.dump(dict(cx=cx.tolist(), cy=cy.tolist(), ic=ic.tolist(), ir=ir.tolist(), SC=SC), open('poly2.json', 'w'))
np.savez('gcp_pairs.npz', pix=P_pix[keep], m=P_m[keep], res=res[keep])
# implied scale & rotation at map centre
c0, r0 = W / 2, H / 2
x0_, y0_ = fwd(np.array([c0]), np.array([r0])); x1_, y1_ = fwd(np.array([c0 + 1000]), np.array([r0]))
print('px per m at centre', 1000 / np.hypot(x1_ - x0_, y1_ - y0_)[0], 'rotation deg', np.degrees(np.arctan2((y1_ - y0_)[0], (x1_ - x0_)[0])))
