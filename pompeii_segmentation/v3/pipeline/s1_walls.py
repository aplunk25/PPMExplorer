import cv2, numpy as np
cv2.setNumThreads(2)
im = cv2.imread('/mnt/user-data/uploads/Downloads/Map3-PompeiiEntrancesLG.png')
b, g, r = im[..., 0].astype(np.int16), im[..., 1].astype(np.int16), im[..., 2].astype(np.int16)
mx = np.maximum(np.maximum(b, g), r); mn = np.minimum(np.minimum(b, g), r)
del b, g, r
# black/dark-gray ink, low saturation (excludes cyan text, red border, light-gray context lines)
dark = (mx < 150) & ((mx - mn) < 60)
del mx, mn, im
dark = dark.astype(np.uint8)
# solidify double-line walls: close tiny gaps between the two outline strokes
k3 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
walls = cv2.morphologyEx(dark, cv2.MORPH_CLOSE, k3)
# fill enclosed white slivers that are wall interiors (small area OR thin)
inv = (1 - walls).astype(np.uint8)
n, lab, st, _ = cv2.connectedComponentsWithStats(inv, connectivity=4)
area = st[:, cv2.CC_STAT_AREA]
w = st[:, cv2.CC_STAT_WIDTH]; h = st[:, cv2.CC_STAT_HEIGHT]
# thinness: area / max(w,h) approximates mean width for elongated slivers
thin = area / np.maximum(np.maximum(w, h), 1)
fill = (area < 25) | ((thin < 3.0) & (area < 3000))
fill[0] = False
walls[(lab > 0) & fill[lab]] = 1
del lab, inv
np.save('walls.npy', walls)
print('components', n, 'filled', fill.sum(), 'wall frac', walls.mean())
