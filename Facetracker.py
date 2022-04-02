import cv2
import mediapipe as mp
from djitellopy import Tello
import time
import numpy as np
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils



def intializeTello():
    myDrone = Tello()
    myDrone.connect()
    myDrone.for_back_velocity = 0
    myDrone.left_right_velocity = 0
    myDrone.up_down_velocity = 0
    myDrone.yaw_velocity = 0
    myDrone.speed = 0
    print(myDrone.get_battery())
    myDrone.streamoff()
    myDrone.streamon()
    return myDrone


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

def telloGetFrame(myDrone, w, h):
    myFrame = myDrone.get_frame_read()
    myFrame = myFrame.frame
    img = cv2.resize(myFrame, (w, h))
    return img

myDrone = intializeTello()
# cap = cv2.VideoCapture(0)
w=640
h=480
pError = [0,0,0]


# cap = cv2.VideoCapture(0)

with mp_face_detection.FaceDetection(
    model_selection=0, min_detection_confidence=0.7) as face_detection:

  flag=True
  while True:
    t=time.time()
    image = telloGetFrame(myDrone, w, h)
    if flag:
      myDrone.takeoff()
      flag=False
    height=myDrone.get_height()
    if height<160:
      myDrone.move_up(170-height)
    h,w,c = image.shape
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    myFacesListC = []
    myFaceListArea = []
    if results.detections:
      for detection in results.detections:
        print(detection)
        # mp_drawing.draw_detection(image, detection)
        location_data = detection.location_data
        if location_data.format == location_data.RELATIVE_BOUNDING_BOX:
          bb = location_data.relative_bounding_box
          bb_box = [
            bb.xmin, bb.ymin,
            bb.width, bb.height,
          ]

          x1, y1 = int((bb_box[0])*w), int((bb_box[1])*h)
          x2, y2 = int((bb_box[0] + bb_box[2])*w), int((bb_box[1] + bb_box[3])*h)

          print(x1,y1,x2,y2)

          cv2.rectangle(image, (x1, y1), (x2, y2), (255,0,0), 2)


          cx = (x1 + x2 ) /2
          cy = (y1 + y2 ) /2
          area = bb_box[2] * bb_box[3]
          myFacesListC.append([cx, cy])
          myFaceListArea.append(area)

      if len(myFaceListArea) != 0:
        i = myFaceListArea.index(max(myFaceListArea))
        info = [myFacesListC[i], myFaceListArea[i]]
      else:
        info = [[0, 0], 0]
      if y2 - y1 > 180 or x2 - x1 > 180:
          # continue
          cv2.putText(image, 'WARNING!!!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
          myDrone.send_rc_control(0, -20, 0, 0)
          # cv2.putText(image, 'Back', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
      elif y2-y1 <180 and x2-x1 < 180:
        pError = trackFace(myDrone, info[0], info[1], w,h, pid, pError,t-time.time())
    cv2.imshow('MediaPipe Face Detection',image)
    if cv2.waitKey(5) & 0xFF == 27:
      myDrone.land()
      myDrone.end()
      break