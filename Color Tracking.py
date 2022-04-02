from matplotlib.pyplot import flag
from djitellopy import Tello
import cv2
import numpy as np
import time

######################################################################
width = 640  # WIDTH OF THE IMAGE
height = 640  # HEIGHT OF THE IMAGE
deadZone =100
######################################################################

startCounter =0

# CONNECT TO TELLO
mytello = Tello()
mytello.connect()
mytello.for_back_velocity = 0
mytello.left_right_velocity = 0
mytello.up_down_velocity = 0
mytello.yaw_velocity = 0
mytello.speed = 0
print(mytello.get_battery())
mytello.streamoff()
mytello.streamon()


pidx = [-0.1,-0.2,-0.05]
pidy=[0.1, 0.4,0.05]
pidz=[0.1, 0.1,0.05]
pid=[pidx,pidy,pidz]
fbRange = [6400, 6800]


def trackFace(tello, center, area, w,h, pid, pError,dt):
  pidx,pidy,pidz=pid[0],pid[1],pid[2]
  pErrorx,pErrory,pErrorz=pError[0],pError[1],pError[2]
  fb = 0
  x, y = center[0], center[1]
  z=area
  errorx =(w / 2)-x
  errory=(h/2)-y
  errorz=np.mean(fbRange)-z
  dx=errorx - pErrorx
  dy=errory - pErrory
  dz=errorz-pErrorz
  yaw = pidx[0] * errorx + (pidx[1] * dx) + pidx[2]*dx/dt 
  height=pidy[0] * errory + (pidy[1] * dy)+ pidy[2]*dy/dt
  fb=pidz[0] * errorz + (pidz[1] * dz)+ pidz[2]*dz/dt
  print(height,yaw,fb)
  yaw = int(np.clip(yaw, -100, 100))
  height=int(np.clip(height,-100,100))
  fb=int(np.clip(fb,-10,10))
  print(height,yaw,fb)
  if z> fbRange[0] and z < fbRange[1]:
    fb = 0
    errorz=0
  if x == 0:
    yaw = 0
    errorx = 0
  if y==0:
    height=0
    errory=0
  tello.send_rc_control(0, fb, height, yaw)
  return [errorx,errory,errorz]

width = 640  # WIDTH OF THE IMAGE
height = 640  # HEIGHT OF THE IMAGE
deadZone =100
pts = [(0,0)]*10
pError = [0,0,0]
ctr = 0
fl=True
while True:
    t=time.time()
    if startCounter == 0:
        mytello.takeoff()
        startCounter = 1

    # success, myFrame = cap.read()
    height=mytello.get_height()
    if height<140 and fl:
                mytello.move_up(150-height)
                fl=False
    myFrame = mytello.get_frame_read().frame
    img = cv2.resize(myFrame, (width, height))
    imgContour = img.copy()
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    greenLower = np.asarray([30, 100, 125])
    greenUpper = np.asarray([179,255, 255])
    mask = cv2.inRange(imgHsv, greenLower, greenUpper)
    mask = cv2.erode(mask, None, iterations=2)
    mask = cv2.dilate(mask, None, iterations=2)

    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None

    # only proceed if at least one contour was found
    if len(cnts) > 0:
        # find the largest contour in the mask, then use
        # it to compute the minimum enclosing circle and
        # centroid
        c = max(cnts, key=cv2.contourArea)
        ((x, y), radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        area = cv2.contourArea(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))

        # only proceed if the radius meets a minimum size
        if radius > 10:
                # draw the circle and centroid on the frame,
                # then update the list of tracked points
                cv2.circle(img, (int(x), int(y)), int(radius),
                        (0, 255, 255), 2)
                cv2.circle(img, center, 5, (0, 0, 255), -1)
        pts[ctr%10] = center
        ctr = ctr+1

    # loop over the set of tracked points
        for i in range(1, len(pts)):
                # if either of the tracked points are None, ignore
                # them
                if pts[i - 1] is None or pts[i] is None:
                        continue
                cv2.line(img, pts[i - 1], pts[i], (0, 0, 255), 1)

        # show the frame to our screen
        pError = trackFace(mytello, center, area, width,height, pid, pError,t-time.time())
    img = cv2.resize(img, (width, height))
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        mytello.land()
        break
