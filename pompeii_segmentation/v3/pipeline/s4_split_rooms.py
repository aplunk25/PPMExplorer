"""Split rooms > SPLIT_M2 into 4 equal-area quarters (oriented to the room's own axes)
and write the room polygons and door openings."""
import cv2, numpy as np, json
PXM = 11.806                    # px per metre (from P-LOD georeference fit)
MIN_AREA = 60
EXTERIOR = 400000
SPLIT_M2 = 100
SPLIT_PX = SPLIT_M2 * PXM ** 2

closed = np.load('closed.npy')
H, W = closed.shape
free = (1 - closed).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(free, connectivity=4, ltype=cv2.CV_32S)
del free
area = st[:, 4]
cl = json.load(open('closures.json'))

nid = np.zeros((H, W), np.int32)
nodes = {}          # node_id -> dict
next_id = 1
split_stats = []
K4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], np.uint8)

def split4(mask):
    ys, xs = np.nonzero(mask)
    c = np.stack([ys, xs], 1).astype(float)
    mu = c.mean(0); cc = c - mu
    w, v = np.linalg.eigh(np.cov(cc.T))
    major, minor = v[:, 1], v[:, 0]
    t1 = cc @ major; t2 = cc @ minor
    h1 = t1 > np.median(t1)
    q = np.zeros(len(ys), np.int8)
    for hv in (False, True):
        sel = h1 == hv
        m2 = np.median(t2[sel])
        q[sel] = (2 if hv else 0) + (t2[sel] > m2)
    qmap = np.full(mask.shape, -1, np.int8); qmap[ys, xs] = q
    # connectivity repair: every quarter must be one 4-connected piece
    for _ in range(20):
        changed = False
        for k in range(4):
            m = (qmap == k).astype(np.uint8)
            nc, cl_ = cv2.connectedComponents(m, connectivity=4)
            if nc <= 2: continue
            sizes = np.bincount(cl_.ravel())[1:]
            keep = 1 + int(np.argmax(sizes))
            for f in range(1, nc):
                if f == keep: continue
                frag = (cl_ == f).astype(np.uint8)
                ring = cv2.dilate(frag, K4) & (1 - frag)
                nb = qmap[ring.astype(bool)]; nb = nb[(nb >= 0) & (nb != k)]
                if len(nb) == 0: continue
                qmap[frag.astype(bool)] = np.bincount(nb, minlength=4).argmax(); changed = True
        if not changed: break
    return qmap

from skimage.graph import MCP_Geometric
def balanced(qmap, mask):
    s = np.bincount(qmap[mask], minlength=4) / mask.sum()
    return s.min() > 0.22 and s.max() < 0.28

def geodesic_split(mask, qmap0):
    """Geodesic power-diagram partition into 4 equal areas, seeded from the PCA quarters."""
    m8 = mask.astype(np.uint8)
    seeds = []
    for k in range(4):
        qk = np.pad((qmap0 == k).astype(np.uint8), 1)
        dt = cv2.distanceTransform(qk, cv2.DIST_L2, 5)[1:-1, 1:-1]
        seeds.append(np.unravel_index(np.argmax(dt), dt.shape))
    cost = np.where(mask, 1.0, np.inf)
    D = []
    for sd in seeds:
        mcp = MCP_Geometric(cost, fully_connected=True)
        d, _ = mcp.find_costs([sd]); D.append(d)
    D = np.stack(D)[:, mask]                       # 4 x N
    N = D.shape[1]; wts = np.zeros(4)
    scale = np.sqrt(N) / 4
    best = None
    for it in range(80):
        lab_ = np.argmin(D - wts[:, None], 0)
        s = np.bincount(lab_, minlength=4) / N
        err = np.abs(s - 0.25).max()
        if best is None or err < best[0]: best = (err, lab_.copy())
        if err < 0.005: break
        wts += (0.25 - s) * scale * (0.9 ** (it / 10))
    q = np.full(mask.shape, -1, np.int8); q[mask] = best[1]
    return q

def fix_conn(qmap):
    for _ in range(20):
        changed = False
        for k in range(4):
            m = (qmap == k).astype(np.uint8)
            nc, cl_ = cv2.connectedComponents(m, connectivity=4)
            if nc <= 2: continue
            sizes = np.bincount(cl_.ravel())[1:]; keep = 1 + int(np.argmax(sizes))
            for f in range(1, nc):
                if f == keep: continue
                frag = (cl_ == f).astype(np.uint8)
                ring = cv2.dilate(frag, K4) & (1 - frag)
                nb = qmap[ring.astype(bool)]; nb = nb[(nb >= 0) & (nb != k)]
                if len(nb): qmap[frag.astype(bool)] = np.bincount(nb, minlength=4).argmax(); changed = True
        if not changed: break
    return qmap

def quarters(mask):
    q = split4(mask)
    if balanced(q, mask): return q, 'axis'
    g = fix_conn(geodesic_split(mask, q))
    sg = np.bincount(g[mask], minlength=4) / mask.sum(); sq = np.bincount(q[mask], minlength=4) / mask.sum()
    return (g, 'geodesic') if np.abs(sg - .25).max() < np.abs(sq - .25).max() else (q, 'axis')

for i in range(1, n):
    a = area[i]
    if a < MIN_AREA: continue
    x, y, w, h, _ = st[i]
    sub = lab[y:y + h, x:x + w] == i
    cls = 'exterior' if a > EXTERIOR else ('fragment' if a < 1.5 * PXM ** 2 else 'room')
    if cls == 'room' and max(w, h) > 200:
        pts = np.argwhere(sub)[:, ::-1].astype(np.float32)
        (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
        lng, sht = max(rw, rh), max(1.0, min(rw, rh))
        # sidewalk / street stretch: long (> ~25 m) and thin, or long and sparse within its box (curved/branching street)
        if lng > 267 and (lng / sht > 6 or (a / (rw * rh) < 0.3 and cv2.distanceTransform(np.pad(sub.astype(np.uint8), 1), cv2.DIST_L2, 3).max() < 40)):
            cls = 'street'
    room_id = i
    if cls == 'room' and a > SPLIT_PX:
        qmap, method = quarters(sub)
        parts = []
        for k in range(4):
            m = qmap == k
            if m.sum() == 0: continue
            nid[y:y + h, x:x + w][m] = next_id
            nodes[next_id] = dict(cls='room_quarter', parent=room_id, quarter=k + 1, bbox=(x, y, w, h), method=method)
            parts.append(int(m.sum())); next_id += 1
        split_stats.append([p / a for p in parts])
    else:
        nid[y:y + h, x:x + w][sub] = next_id
        nodes[next_id] = dict(cls=cls, parent=room_id, quarter=0, bbox=(x, y, w, h))
        next_id += 1
del lab
ss = np.array([s for s in split_stats if len(s) == 4])
import collections; print(collections.Counter(d.get('method') for d in nodes.values() if d['quarter']==1)); print('nodes', len(nodes), 'split rooms', len(split_stats), 'quarter share min/mean/max',
      ss.min().round(3), ss.mean().round(3), ss.max().round(3), 'within 22-28%:', ((ss > .22) & (ss < .28)).all(1).mean().round(3))
np.save('nid.npy', nid)

# ---- node geometry: polygons + representative point (pole of inaccessibility) ----
def to_xy(px, py): return [round(px + 0.5, 1), round(-(py + 0.5), 1)]
room_feats = []
for k, d in nodes.items():
    x, y, w, h = d['bbox']
    x0, y0 = max(0, x - 1), max(0, y - 1)
    m = (nid[y0:y + h + 1, x0:x + w + 1] == k).astype(np.uint8)
    a = int(m.sum())
    dt = cv2.distanceTransform(np.pad(m, 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    py, px = np.unravel_index(np.argmax(dt), dt.shape)
    rp = to_xy(px + x0, py + y0)
    cs, hier = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    hier = hier[0]; polys = []
    for ci, c in enumerate(cs):
        if hier[ci][3] != -1: continue
        def ring(cc):
            cc = cv2.approxPolyDP(cc, 0.9, True)[:, 0, :]
            pts = [to_xy(px_ + x0, py_ + y0) for px_, py_ in cc]
            if len(pts) < 3: return None
            return pts + [pts[0]]
        r0 = ring(c)
        if r0 is None: continue
        rings = [r0]; ch = hier[ci][2]
        while ch != -1:
            if cv2.contourArea(cs[ch]) >= 20:
                r = ring(cs[ch])
                if r: rings.append(r)
            ch = hier[ch][0]
        polys.append(rings)
    d['area'] = a; d['rp'] = rp
    props = dict(node_id=k, parent_room_id=d['parent'], quarter=d['quarter'], node_class=d['cls'],
                 split_method=d.get('method', ''),
                 area_px=a, area_m2_est=round(a / PXM ** 2, 2), rep_x=rp[0], rep_y=rp[1],
                 max_inscribed_m=round(float(dt.max()) / PXM, 2))
    if polys:
        room_feats.append({'type': 'Feature', 'properties': props, 'geometry': {'type': 'MultiPolygon', 'coordinates': polys}})

# ---- openings: each sealed doorway with the room on either side ----
def side(yy, xx):
    yy, xx = int(round(yy)), int(round(xx))
    if 0 <= yy < H and 0 <= xx < W:
        v = int(nid[yy, xx]); return v or None
    return None
open_feats = []; did = 0
for a_, b_, c_, d_, kind in cl:
    dy, dx = c_ - a_, d_ - b_; L = float(np.hypot(dy, dx))
    if L < 1: continue
    ny, nx = -dx / L, dy / L; my, mx = (a_ + c_) / 2, (b_ + d_) / 2
    ra = rb = None
    for off in (4, 6, 9):
        ra = ra or side(my + off * ny, mx + off * nx); rb = rb or side(my - off * ny, mx - off * nx)
    if ra is None or rb is None or ra == rb: continue
    did += 1
    g = [to_xy(b_, a_), to_xy(d_, c_)]
    open_feats.append({'type': 'Feature', 'properties': dict(door_id=did, node_a=ra, node_b=rb, closure_rule=kind,
                      width_px=round(L, 1), width_m_est=round(L / PXM, 2)), 'geometry': {'type': 'LineString', 'coordinates': g}})
print('rooms', len(room_feats), 'openings', len(open_feats))

for name, fs in [('rooms_v2', room_feats), ('openings_v2', open_feats)]:
    json.dump({'type': 'FeatureCollection', 'name': name, 'features': fs}, open(f'out/{name}.geojson', 'w'))
