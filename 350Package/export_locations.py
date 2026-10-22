"""Write embeddings/random_samples/locations.json: image_id -> location name, scope and centroid.

Reads the Cosmos query exports in data/queries/, whose location field is a Python-literal string.
Usage: python3 350Package/export_locations.py
"""

import ast
import csv
import glob
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
METADATA = HERE / 'embeddings/random_samples/metadata.csv'
QUERIES = HERE / 'data/queries'
OUT = HERE / 'embeddings/random_samples/locations.json'

with METADATA.open(encoding='utf-8-sig') as f:
    ids = [row['image_id'] for row in csv.DictReader(f)]
wanted = set(ids)

locations = {}
for path in sorted(glob.glob(str(QUERIES / '*.json'))):
    for record in json.loads(Path(path).read_text(encoding='utf-8')):
        if record['id'] not in wanted or record['id'] in locations:
            continue
        location = record.get('location')
        if isinstance(location, str):
            location = ast.literal_eval(location)
        props = ((location or {}).get('geojson') or {}).get('properties') or {}
        centroid = props.get('centroid')
        if not centroid:
            continue
        locations[record['id']] = {
            'name': props.get('name', ''),
            'scope': props.get('scope', ''),
            'lat': round(centroid['lat'], 7),
            'lon': round(centroid['lon'], 7),
        }

OUT.write_text(json.dumps({i: locations[i] for i in ids if i in locations}, indent=1))
print(f'{len(locations)}/{len(ids)} samples located -> {OUT}')
