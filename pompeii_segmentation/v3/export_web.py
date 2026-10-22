"""Export a trimmed copy of rooms_v3_wgs84.geojson for the web map (static/data/).

Drops exterior/fragment polygons, keeps a few properties, and rounds coordinates to 7 decimals (~1 cm).
Usage: python3 pompeii_segmentation/v3/export_web.py
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SRC = HERE / 'rooms_v3_wgs84.geojson'
DST = HERE.parents[1] / 'static/data/pompeii_rooms_v3.geojson'
KEEP_CLASSES = {'room', 'room_quarter', 'street'}
KEEP_PROPS = ('node_id', 'parent_room_id', 'node_class', 'area_m2')


def round_coords(c):
    if isinstance(c[0], (int, float)):
        return [round(c[0], 7), round(c[1], 7)]
    return [round_coords(x) for x in c]


src = json.loads(SRC.read_text())
features = [
    {
        'type': 'Feature',
        'properties': {k: f['properties'][k] for k in KEEP_PROPS},
        'geometry': {
            'type': f['geometry']['type'],
            'coordinates': round_coords(f['geometry']['coordinates']),
        },
    }
    for f in src['features']
    if f['properties']['node_class'] in KEEP_CLASSES
]

DST.parent.mkdir(parents=True, exist_ok=True)
DST.write_text(json.dumps({'type': 'FeatureCollection', 'features': features}, separators=(',', ':')))
print(f'{len(features)} features -> {DST} ({DST.stat().st_size / 1e6:.1f} MB)')
