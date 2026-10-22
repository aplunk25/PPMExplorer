import cv2, numpy as np, json, sys
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

LMAX = float(sys.argv[1]) if len(sys.argv) > 1 else 45   # max ray door width (px, ~11 px/m)
PMAX = float(sys.argv[2]) if len(sys.argv) > 2 else 55   # max open-front pair width
walls = np.load('walls3.npy')
H, W = walls.shape
k3 = np.ones((3, 3), np.uint8)
thick = cv2.morphologyEx(walls, cv2.MORPH_OPEN, k3)
# drop tiny thick specks (column dots are fine to keep as obstacles, but no endpoints from them)
# skeleton in tiles
skel = np.zeros_like(walls)
T, O = 4000, 40
for y in range(0, H, T):
    for x in range(0, W, T):
        y0, x0 = max(0, y - O), max(0, x - O)
        y1, x1 = min(H, y + T + O), min(W, x + T + O)
        s = skeletonize(thick[y0:y1, x0:x1].astype(bool)).astype(np.uint8)
        skel[y:min(H, y + T), x:min(W, x + T)] = s[y - y0:y - y0 + min(T, H - y), x - x0:x - x0 + min(T, W - x)]
np.save('skel.npy', skel)
nb = cv2.filter2D(skel, cv2.CV_8U, np.ones((3, 3), np.float32), borderType=cv2.BORDER_CONSTANT) - skel
ends = np.argwhere((skel == 1) & (nb == 1))
print('endpoints', len(ends))

OFF = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
def trace(p, maxlen=14):
    path = [tuple(p)]; prev = None; cur = tuple(p)
    for _ in range(maxlen):
        nxt = []
        for dy, dx in OFF:
            q = (cur[0] + dy, cur[1] + dx)
            if 0 <= q[0] < H and 0 <= q[1] < W and skel[q] and q != prev and q not in path:
                nxt.append(q)
        if len(nxt) != 1:
            return path, len(nxt) > 1
        prev, cur = cur, nxt[0]; path.append(cur)
    return path, False

E = []  # (y, x, dy, dx)
for p in ends:
    path, junc = trace(p)
    if len(path) < 6:      # corner spur / speck
        continue
    a = np.array(path[min(len(path) - 1, 12)], float); b = np.array(path[0], float)
    d = b - a; n = np.hypot(*d)
    if n < 1e-6: continue
    d = d / n
    # local PCA of the wall around the end: must be an elongated straight stub aligned with d
    y0, x0 = int(b[0]) - 10, int(b[1]) - 10
    win = thick[max(0, y0):y0 + 21, max(0, x0):x0 + 21]
    pts = np.argwhere(win).astype(float)
    if len(pts) < 10: continue
    pts -= pts.mean(0)
    w_, v_ = np.linalg.eigh(np.cov(pts.T))
    ax = v_[:, 1]
    if w_[1] < 2.5 * max(w_[0], 1e-6) or abs(ax @ d) < 0.9: continue
    if ax @ d < 0: ax = -ax
    E.append((b[0], b[1], ax[0], ax[1]))
E = np.array(E)
print('usable endpoints', len(E)); np.save('E.npy', E)

closures = []
def cast(y, x, dy, dx, lmax):
    # walk out of own wall, then through free space until hitting wall
    t = 0.0; inside = True; start = None
    while t < lmax + 12:
        yy, xx = int(round(y + dy * t)), int(round(x + dx * t))
        if not (0 <= yy < H and 0 <= xx < W): return None
        wv = walls[yy, xx]
        if inside:
            if not wv:
                inside = False; start = (yy, xx); t0 = t
            elif t > 10: return None
        else:
            if wv:
                if t - t0 > lmax: return None
                if not thick[yy, xx]: return None
                return start, (yy, xx), t - t0
        t += 0.5
    return None

for y, x, dy, dx in E:
    best = None
    for ang in (0, 4, -4, 8, -8):
        c, s = np.cos(np.radians(ang)), np.sin(np.radians(ang))
        ddy, ddx = dy * c - dx * s, dy * s + dx * c
        r = cast(y, x, ddy, ddx, LMAX)
        if r is not None:
            best = r; break
    if best: closures.append((best[0], best[1], 'ray'))

# open-front pairs: parallel wall ends side by side
tree = cKDTree(E[:, :2])
pairs = tree.query_pairs(PMAX)
def line_free(p, q):
    n = int(np.hypot(q[0] - p[0], q[1] - p[1])) + 1
    ys = np.linspace(p[0], q[0], n).round().astype(int); xs = np.linspace(p[1], q[1], n).round().astype(int)
    v = walls[ys, xs]
    # ignore the bits inside the two wall ends
    k = 0
    while k < n and v[k]: k += 1
    j = n - 1
    while j >= 0 and v[j]: j -= 1
    if k > 10 or (n - 1 - j) > 10 or j <= k: return None
    if v[k:j + 1].any(): return None
    return (ys[k], xs[k]), (ys[j], xs[j]), j - k
np_ = 0
for i, j in pairs:
    a, b = E[i], E[j]
    if a[2] * b[2] + a[3] * b[3] < 0.85: continue
    u = b[:2] - a[:2]; L = np.hypot(*u); u /= L
    if abs(u[0] * a[2] + u[1] * a[3]) > 0.35: continue
    r = line_free(a[:2], b[:2])
    if r is None or r[2] > PMAX: continue
    closures.append((r[0], r[1], 'pair')); np_ += 1
print('closures', len(closures), 'pairs', np_)

closed = walls.copy()
for p, q, kind in closures:
    cv2.line(closed, (int(p[1]), int(p[0])), (int(q[1]), int(q[0])), 1, 3)
np.save('closed.npy', closed)
json.dump([[int(p[0]), int(p[1]), int(q[0]), int(q[1]), k] for p, q, k in closures], open('closures.json', 'w'))
