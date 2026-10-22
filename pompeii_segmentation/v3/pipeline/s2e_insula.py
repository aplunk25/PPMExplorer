"""Cut every remaining street leak at the insula (block) outline: blocks = thick walls closed by K px
(bridges entrances/shop fronts narrower than K, keeps wider streets open), holes filled.
Free space of the street/exterior network that lies inside a block is cut off along the block outline."""
import cv2, numpy as np, json, sys
K = int(sys.argv[1]) if len(sys.argv) > 1 else 31
closed = np.load('closed.npy'); walls = np.load('walls3.npy')
H, W = closed.shape
thick = cv2.morphologyEx(walls, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
blk = cv2.morphologyEx(thick, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (K, K)))
del thick
# fill holes: background reachable from border stays 0
inv = (1 - blk).astype(np.uint8)
n, lab = cv2.connectedComponents(inv, connectivity=4)
border = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
outside = np.isin(lab, border[border > 0])
del lab, inv
insula = (~outside).astype(np.uint8); del outside, blk
# exterior network = big free components
free = (1 - closed).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4, ltype=cv2.CV_32S)
ext = (st[:, 4] > 400000); ext[0] = False
extm = ext[lab].astype(np.uint8); del lab
grad = insula - cv2.erode(insula, np.ones((3, 3), np.uint8))
cut = (grad == 1) & (extm == 1)
print('cut px', int(cut.sum()))
closed[cut] = 1
# validate: each piece split off the exterior must be house-like (not a street stretch, not huge open ground)
free2 = (1 - closed).astype(np.uint8)
n2, lab2, st2, _ = cv2.connectedComponentsWithStats(free2, connectivity=4, ltype=cv2.CV_32S); del free2
cnt = np.bincount(lab2[extm == 1], minlength=n2)          # pixels formerly exterior per new comp
bad = np.zeros(n2, bool); nacc = nrej = 0
for i in np.nonzero(cnt)[0]:
    if i == 0 or st2[i, 4] > 400000: continue           # still the main exterior
    x, y, w, h, a = st2[i]
    pts = np.argwhere(lab2[y:y + h, x:x + w] == i)[:, ::-1].astype(np.float32)
    (cx, cy), (rw, rh), _ = cv2.minAreaRect(pts)
    asp = max(rw, rh) / max(1.0, min(rw, rh))
    if a > 200000 or asp > 4.0 or a / max(1.0, rw * rh) < 0.25: bad[i] = True; nrej += 1
    else: nacc += 1
badm = bad[lab2].astype(np.uint8); del lab2
undo = cut & (cv2.dilate(badm, np.ones((3, 3), np.uint8)) == 1)
closed[undo] = 0; cut &= ~undo
print('pieces accepted', nacc, 'rejected', nrej, 'cut px kept', int(cut.sum()))
np.save('closed.npy', closed); np.save('insula.npy', insula)
# record cut segments as openings: connected pieces of the cut, endpoints = farthest pair
nc, cl_, cst, _ = cv2.connectedComponentsWithStats(cut.astype(np.uint8), connectivity=8)
segs = []
for i in range(1, nc):
    x, y, w, h, a = cst[i]
    if a < 4: continue
    ys, xs = np.nonzero(cl_[y:y + h, x:x + w] == i)
    pts = np.stack([ys + y, xs + x], 1)
    c = pts.mean(0); d = np.linalg.eigh(np.cov((pts - c).T))[1][:, 1]
    t = (pts - c) @ d
    p, q = pts[t.argmin()], pts[t.argmax()]
    segs.append([int(p[0]), int(p[1]), int(q[0]), int(q[1]), 'insula'])
cl = json.load(open('closures.json')); cl += segs; json.dump(cl, open('closures.json', 'w'))
print('insula cuts', len(segs))
