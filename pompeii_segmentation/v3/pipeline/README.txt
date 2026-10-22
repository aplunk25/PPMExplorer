Pompeii room segmentation, v2. Run in order:
  python3 s1_walls.py            # black ink mask (drops cyan labels, red frame, grey context lines)
  python3 s1c.py                 # solidify double-line walls
  python3 s2_close.py            # doorways: straight wall-end rays + parallel open-front pairs
  python3 s2b_corners.py         # openings between facing convex corners (L-jambs)
  python3 s2c_pockets.py         # shop fronts: cuts that enclose a room-sized pocket
  python3 s2c_pockets.py 400000 200000 700   # same, wider search on the street network (neck rule)
  python3 s2d_necks.py 400000 200000 700     # shortest corner-to-wall necks into the street network
  python3 s2e_insula.py 21       # cut any remaining leak at the insula outline (house-like pieces only)
  python3 s4_split_rooms.py      # quarter rooms >100 m2, write out/rooms_v2.geojson + out/openings_v2.geojson

Room classes (node_class): room, room_quarter (parent_room_id + quarter 1-4), street (sidewalk / street stretch),
exterior (main street network + open ground), fragment (<1.5 m2: niches, wall voids, hatching - filter for embeddings).
Coordinates: pixel space of Map3-PompeiiEntrancesLG.png as QGIS shows it (x = col, y = -row). 11.81 px/m (fitted).

Georeferencing (v3), after s4:
  g1_coarse.py   # scale/rotation/translation search: insula blocks vs P-LOD insula polygons (plod_reference.geojson)
  g2_refine.py   # match P-LOD room polygons to segmented rooms (IoU), robust 2nd-order polynomial pixel->ground
  g3_apply.py    # transform rooms/openings to WGS84, true areas/widths, write GCP .points file
Result: 1786 room-centroid GCPs, residual median 0.10 m, RMS 0.31 m (vs P-LOD).
