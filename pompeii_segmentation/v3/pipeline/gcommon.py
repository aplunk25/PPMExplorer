import json, numpy as np
LAT0, LON0 = 40.7505, 14.4880
MLAT, MLON = 111049.0, 84456.0
def ll2m(lon, lat): return (np.asarray(lon) - LON0) * MLON, (np.asarray(lat) - LAT0) * MLAT
def m2ll(x, y): return np.asarray(x) / MLON + LON0, np.asarray(y) / MLAT + LAT0
def coarse_affine():
    c = json.load(open('coarse.json')); M = np.array(c['M']); D = c['D']
    # x_m = xmin + (M @ [col/D,row/D,1])[0] + ix ; y_m = ymax - ((M@..)[1] + iy)
    A = np.zeros((2, 3))
    A[0, :2] = M[0, :2] / D; A[0, 2] = M[0, 2] + c['ix'] + c['xmin']
    A[1, :2] = -M[1, :2] / D; A[1, 2] = c['ymax'] - (M[1, 2] + c['iy'])
    return A
def load_plod(scopes=('room',)):
    F = json.load(open('plod_reference.geojson'))['features']; out = []
    for f in F:
        if f['properties']['scope'] not in scopes: continue
        g = f['geometry']; rings = [g['coordinates'][0]] if g['type'] == 'Polygon' else [p[0] for p in g['coordinates']]
        for r in rings:
            a = np.array(r); x, y = ll2m(a[:, 0], a[:, 1]); out.append((f['properties'], np.stack([x, y], 1)))
    return out
