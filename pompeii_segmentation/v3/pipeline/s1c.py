import cv2, numpy as np
walls=np.load('walls.npy')
inv=(1-walls).astype(np.uint8)
dt=cv2.distanceTransform(inv,cv2.DIST_L2,5)
core=(dt>2.5).astype(np.uint8); del dt
free=cv2.dilate(core,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5)))&inv
del core,inv
walls=(1-free).astype(np.uint8)
np.save('walls3.npy',walls)
print(walls.mean())
cv2.imwrite('w2.png',cv2.resize(255-walls[4700:4950,9800:10150]*255,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST))
cv2.imwrite('w1.png',255-walls[4500:5100,9300:10300]*255)
