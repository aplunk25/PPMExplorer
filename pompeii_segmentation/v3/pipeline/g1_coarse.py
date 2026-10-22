"""Coarse georeference: brute-force scale/rotation + FFT translation matching of our insula
blocks against P-LOD insula polygons (both rasterised at 1 m)."""
import json, numpy as np, cv2
LAT0, LON0 = 40.7505, 14.4880
MLAT, MLON = 111049.0, 84456.0
def ll2m(lon, lat): return (np.asarray(lon) - LON0) * MLON, (np.asarray(lat) - LAT0) * MLAT

F = json.load(open('plod_reference.geojson'))['features']
polys = []
for f in F:
    if f['properties']['scope'] != 'insula': continue
    g = f['geometry']; rings = [g['coordinates'][0]] if g['type'] == 'Polygon' else [p[0] for p in g['coordinates']]
    for r in rings:
        a = np.array(r); x, y = ll2m(a[:, 0], a[:, 1]); polys.append(np.stack([x, y], 1))
allp = np.concatenate(polys)
PAD = 150
xmin, ymin = allp.min(0) - PAD; xmax, ymax = allp.max(0) + PAD
Wm, Hm = int(xmax - xmin), int(ymax - ymin)
ref = np.zeros((Hm, Wm), np.float32)
for p in polys:
    q = np.stack([p[:, 0] - xmin, ymax - p[:, 1]], 1).round().astype(np.int32)   # row = north->south
    cv2.fillPoly(ref, [q], 1.0)
print('ref grid', ref.shape, 'filled', ref.sum())

ins = np.load('insula.npy')
H, W = ins.shape
D = 4
small = cv2.resize(ins.astype(np.float32), (W // D, H // D), interpolation=cv2.INTER_AREA)
refz = ref - ref.mean()
best = []
for s in np.arange(9.0, 12.6, 0.15):            # px per metre
    for rot in np.arange(-8, 8.1, 1.0):           # degrees, map rotation relative to north
        # small-image px -> metric grid px (1 m): scale D/s, rotate
        k = D / s
        M = cv2.getRotationMatrix2D((0, 0), rot, k)
        corners = np.array([[0, 0], [W // D, 0], [0, H // D], [W // D, H // D]], float)
        cc = corners @ M[:, :2].T
        M[:, 2] -= cc.min(0)
        size = tuple((cc.max(0) - cc.min(0)).astype(int) + 1)
        warped = cv2.warpAffine(small, M, size)
        # correlate warped (template) over ref via FFT
        h2, w2 = warped.shape
        Hh, Ww = max(Hm, h2) + h2, max(Wm, w2) + w2
        A = np.fft.rfft2(refz, (Hh, Ww)); B = np.fft.rfft2(warped - warped.mean(), (Hh, Ww))
        c = np.fft.irfft2(A * np.conj(B), (Hh, Ww))
        iy, ix = np.unravel_index(np.argmax(c), c.shape)
        if iy > Hh // 2: iy -= Hh
        if ix > Ww // 2: ix -= Ww
        iy, ix = int(iy), int(ix)
        c = {(iy, ix): c.max()}
        best.append((c[(iy, ix)], s, rot, iy, ix, M.copy()))
best.sort(key=lambda b: -b[0])
for b in best[:6]: print(round(b[0], 1), round(b[1], 2), b[2], b[3], b[4])
score, s, rot, iy, ix, M = best[0]
# compose: full-res pixel (col,row) -> small (col/D,row/D) -> warped (M) -> ref grid (+ix, +iy) -> metric
Hh = None
def pix2m(col, row):
    p = np.stack([np.asarray(col, float) / D, np.asarray(row, float) / D, np.ones(np.size(col))], 0)
    w = M @ p
    gx, gy = w[0] + ix, w[1] + iy
    return gx + xmin, ymax - gy
json.dump(dict(s=s, rot=rot, iy=int(iy), ix=int(ix), M=M.tolist(), xmin=float(xmin), ymax=float(ymax), D=D), open('coarse.json', 'w'))
# sample check: affine params
for c_, r_ in [(0, 0), (W, 0), (0, H), (W, H)]:
    print((c_, r_), [round(float(v), 1) for v in pix2m([c_], [r_])])
