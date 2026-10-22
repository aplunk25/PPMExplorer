import json, numpy as np
from gcommon import *
P = json.load(open('poly2.json')); cx, cy = np.array(P['cx']), np.array(P['cy']); SC = P['SC']
def feats(c, r):
    c = np.asarray(c, float) / SC; r = np.asarray(r, float) / SC
    return np.stack([np.ones_like(c), c, r, c * c, c * r, r * r], -1)
def xy2m(X, Y):  # our pixel-space vector coords (x=col+0.5, y=-(row+0.5)) -> local metres
    f = feats(np.asarray(X) - 0.5, -np.asarray(Y) - 0.5); return f @ cx, f @ cy
def xy2ll(X, Y):
    x, y = xy2m(X, Y); lon, lat = m2ll(x, y); return lon, lat
def ring_area_m(ring):
    a = np.array(ring); x, y = xy2m(a[:, 0], a[:, 1]); return 0.5 * (x[:-1] * y[1:] - x[1:] * y[:-1]).sum()

R = json.load(open('out/rooms_v2.geojson'))
for f in R['features']:
    polys = f['geometry']['coordinates']; area = 0.0; newp = []
    for rings in polys:
        nr = []
        for j, ring in enumerate(rings):
            a = abs(ring_area_m(ring)); area += a if j == 0 else -a
            a_ = np.array(ring); lon, lat = xy2ll(a_[:, 0], a_[:, 1])
            nr.append([[round(float(u), 8), round(float(v), 8)] for u, v in zip(lon, lat)])
        newp.append(nr)
    f['geometry']['coordinates'] = newp
    p = f['properties']
    lon, lat = xy2ll(p.pop('rep_x'), p.pop('rep_y')); p['rep_lon'] = round(float(lon), 8); p['rep_lat'] = round(float(lat), 8)
    p.pop('area_px', None); p.pop('area_m2_est', None)
    p['area_m2'] = round(area, 2)
O = json.load(open('out/openings_v2.geojson'))
for f in O['features']:
    a = np.array(f['geometry']['coordinates']); lon, lat = xy2ll(a[:, 0], a[:, 1])
    f['geometry']['coordinates'] = [[round(float(u), 8), round(float(v), 8)] for u, v in zip(lon, lat)]
    x, y = xy2m(a[:, 0], a[:, 1]); p = f['properties']
    p.pop('width_px', None); p.pop('width_m_est', None); p['width_m'] = round(float(np.hypot(x[1] - x[0], y[1] - y[0])), 2)
for d, name in [(R, 'rooms_v3_wgs84'), (O, 'openings_v3_wgs84')]:
    d['name'] = name; d['crs'] = {'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:OGC:1.3:CRS84'}}
    json.dump(d, open(f'geo/{name}.geojson', 'w'))
# GCPs: matched room centroids (inliers). pix are cv2 index coords -> GDAL pixel/line = +0.5
G = np.load('gcp_pairs.npz'); pix = G['pix']; lon, lat = m2ll(G['m'][:, 0], G['m'][:, 1])
with open('geo/Map3-PompeiiEntrancesLG.png.points', 'w') as fh:
    fh.write('#CRS: GEOGCRS["WGS 84",DATUM["World Geodetic System 1984",ELLIPSOID["WGS 84",6378137,298.257223563]],CS[ellipsoidal,2],AXIS["geodetic latitude (Lat)",north],AXIS["geodetic longitude (Lon)",east],ANGLEUNIT["degree",0.0174532925199433],ID["EPSG",4326]]\n')
    fh.write('mapX,mapY,sourceX,sourceY,enable,dX,dY,residual\n')
    for (c, r), u, v in zip(pix, lon, lat):
        fh.write(f'{u:.9f},{v:.9f},{c + 0.5:.3f},{-(r + 0.5):.3f},1,0,0,0\n')
json.dump([[float(c + 0.5), float(r + 0.5), float(u), float(v)] for (c, r), u, v in zip(pix, lon, lat)], open('geo/gcps.json', 'w'))
a = np.array([f['properties']['area_m2'] for f in R['features'] if f['properties']['node_class'] in ('room', 'room_quarter')])
print('rooms', len(R['features']), 'median room area m2', np.median(a), 'gcps', len(pix))
