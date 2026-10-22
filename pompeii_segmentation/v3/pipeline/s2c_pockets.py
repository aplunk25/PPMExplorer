"""Seal wide openings (shop fronts, wide fauces) that leak rooms into the street/exterior network.
A candidate cut is accepted only if it encloses a room-sized pocket off the giant component."""
import cv2, numpy as np, json, sys
GMIN=int(sys.argv[1]) if len(sys.argv)>1 else 25000
PMAXA=int(sys.argv[2]) if len(sys.argv)>2 else 60000
RW=int(sys.argv[3]) if len(sys.argv)>3 else 260
from scipy.spatial import cKDTree
closed = np.load('closed.npy'); walls = np.load('walls3.npy')
H, W = closed.shape
E = np.load('E.npy'); P = np.load('P.npy'); B = np.load('B.npy')
free = (1 - closed).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4, ltype=cv2.CV_32S)
giant = st[:, 4] > GMIN; giant[0] = False
print('giant comps', giant.sum())

def seg_free(p, q, margin=3):
    L = np.hypot(q[0] - p[0], q[1] - p[1]); m = int(L) + 1
    ys = np.linspace(p[0], q[0], m).round().astype(int); xs = np.linspace(p[1], q[1], m).round().astype(int)
    v = closed[ys, xs]
    k = 0
    while k < m and v[k]: k += 1
    j = m - 1
    while j >= 0 and v[j]: j -= 1
    if k > 10 or (m - 1 - j) > 10 or j - k < 5: return None
    if v[k:j + 1].any(): return None
    return (ys[k], xs[k]), (ys[j], xs[j]), j - k

cands = []
# (a) convex corner pairs
tree = cKDTree(P)
for i, j in tree.query_pairs(80):
    p, q = P[i], P[j]; u = q - p; L = np.hypot(*u)
    if L < 30: continue
    u = u / L
    if B[i] @ u < 0.2 or B[j] @ (-u) < 0.2: continue
    cands.append((L, p, q))
# (b) parallel wall-end pairs
tree = cKDTree(E[:, :2])
for i, j in tree.query_pairs(85):
    a, b = E[i], E[j]
    if a[2] * b[2] + a[3] * b[3] < 0.8: continue
    u = b[:2] - a[:2]; L = np.hypot(*u); u = u / L
    if abs(u @ a[2:]) > 0.4: continue
    cands.append((L, a[:2], b[:2]))
# (c) endpoint-to-corner (pier end to facing L-corner)
tc = cKDTree(P)
for i, js in enumerate(tc.query_ball_point(E[:, :2], 80)):
    a = E[i]
    for j in js:
        u = P[j] - a[:2]; L = np.hypot(*u)
        if L < 20: continue
        u = u / L
        if abs(u @ a[2:]) > 0.4 or B[j] @ (-u) < 0.2: continue
        cands.append((L, a[:2], P[j]))
cands.sort(key=lambda c: c[0])
print('candidates', len(cands))

R = RW
acc = []
used = set()
for L, p, q in cands:
    r = seg_free(p, q)
    if r is None: continue
    s, e, ln = r
    my, mx = (s[0] + e[0]) // 2, (s[1] + e[1]) // 2
    gl = lab[my, mx]
    if not giant[gl]: continue
    ks, ke = (int(p[0]) // 4, int(p[1]) // 4), (int(q[0]) // 4, int(q[1]) // 4)
    if ks in used or ke in used: continue
    y0, y1, x0, x1 = max(0, my - R), min(H, my + R), max(0, mx - R), min(W, mx + R)
    win = (1 - closed[y0:y1, x0:x1]).astype(np.uint8)
    cv2.line(win, (s[1] - x0, s[0] - y0), (e[1] - x0, e[0] - y0), 0, 3)
    nl, wl, ws, _ = cv2.connectedComponentsWithStats(win, connectivity=4)
    dy, dx = e[0] - s[0], e[1] - s[1]; nn = np.hypot(dy, dx); ny, nx = -dx / nn, dy / nn
    sides = []
    for sg in (1, -1):
        yy, xx = int(round(my + sg * 5 * ny)) - y0, int(round(mx + sg * 5 * nx)) - x0
        if 0 <= yy < win.shape[0] and 0 <= xx < win.shape[1]: sides.append(wl[yy, xx])
    if len(sides) != 2 or 0 in sides or sides[0] == sides[1]: continue
    ok = None
    for l in sides:
        x, y, w, h, a = ws[l]
        enclosed = x > 0 and y > 0 and x + w < win.shape[1] and y + h < win.shape[0]
        if enclosed and a >= 150 and a < min(PMAXA, 0.6 * st[gl, 4]) and a > 0.5 * ln * ln:
            if a > 60000 or R > 300:
                # large pocket: must be entered through a door-like neck, not a full-width street cut
                pm = np.pad((wl[y:y + h, x:x + w] == l).astype(np.uint8), 1)
                if ln > 1.4 * cv2.distanceTransform(pm, cv2.DIST_L2, 5).max(): continue
                if ln > 40: continue
            ok = l
    if ok is None: continue
    used.add(ks); used.add(ke)
    sub = lab[y0:y1, x0:x1]; sub[wl == ok] = 0
    cv2.line(closed, (int(s[1]), int(s[0])), (int(e[1]), int(e[0])), 1, 3)
    acc.append([int(s[0]), int(s[1]), int(e[0]), int(e[1]), 'facade'])
print('accepted', len(acc))
np.save('closed.npy', closed)
cl = json.load(open('closures.json')); cl += acc
json.dump(cl, open('closures.json', 'w'))
