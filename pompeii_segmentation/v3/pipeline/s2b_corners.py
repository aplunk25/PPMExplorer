import cv2, numpy as np, json, sys
from scipy.spatial import cKDTree
CMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 36
walls = np.load('walls3.npy'); closed = np.load('closed.npy')
H, W = walls.shape
k3 = np.ones((3, 3), np.uint8)
thick = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k3)
cs, _ = cv2.findContours(thick, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
K = 5
P, B = [], []
for c in cs:
    c = c[:, 0, :]
    n = len(c)
    if n < 4 * K: continue
    a = np.roll(c, K, 0).astype(float); b = np.roll(c, -K, 0).astype(float); p = c.astype(float)
    v1 = a - p; v2 = b - p
    cosang = (v1 * v2).sum(1) / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1) + 1e-9)
    m = (a + b) / 2
    mi = np.clip(m.round().astype(int), 0, [W - 1, H - 1])
    convex = thick[mi[:, 1], mi[:, 0]] == 1
    cand = (cosang > -0.45) & convex      # angle < ~117 deg
    # non-max suppression along contour: keep local max of cosang within +-K
    score = np.where(cand, cosang, -2)
    lm = np.ones(n, bool)
    for s in range(1, K + 1):
        lm &= (score >= np.roll(score, s)) & (score > np.roll(score, -s))
    idx = np.where(cand & lm)[0]
    for i in idx:
        bis = p[i] - m[i]; nb = np.linalg.norm(bis)
        if nb < 1e-6: continue
        P.append((p[i][1], p[i][0])); B.append((bis[1] / nb, bis[0] / nb))  # (y,x)
P = np.array(P); B = np.array(B)
print('convex corners', len(P)); np.save('P.npy', P); np.save('B.npy', B)
tree = cKDTree(P)
pairs = tree.query_pairs(CMAX)
cand = []
for i, j in pairs:
    p, q = P[i], P[j]
    u = q - p; L = np.hypot(*u)
    if L < 10: continue
    u /= L
    if B[i] @ u < 0.2 or B[j] @ (-u) < 0.2: continue
    n = int(L) + 1
    ys = np.linspace(p[0], q[0], n).round().astype(int); xs = np.linspace(p[1], q[1], n).round().astype(int)
    inner = closed[ys[3:-3], xs[3:-3]]
    if inner.any(): continue
    cand.append((L, i, j))
cand.sort()
# mutual nearest: each corner used once, shortest first
used = set(); out = []
best = {}
for L, i, j in cand:
    best.setdefault(i, (L, j)); best.setdefault(j, (L, i))
for L, i, j in cand:
    if i in used or j in used: continue
    if best[i][1] != j and best[j][1] != i: continue
    used.add(i); used.add(j)
    out.append((P[i], P[j]))
print('corner closures', len(out))
cl = json.load(open('closures.json'))
for p, q in out:
    cv2.line(closed, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, 3)
    cl.append([int(p[0]), int(p[1]), int(q[0]), int(q[1]), 'corner'])
np.save('closed.npy', closed)
json.dump(cl, open('closures.json', 'w'))
