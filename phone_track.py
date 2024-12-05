import cv2
from ultralytics import YOLO,solutions 
import numpy as np
import requests
import cv2
from ultralytics import YOLO,solutions 

def count_objects_in_region(model_path,fullscreen=False):
    """Count objects in a specific region within a video"""

    model=YOLO(model_path)
   

    
    line_points = [(20, 400), (1080, 400)]

    counter=solutions.ObjectCounter(
        view_img=False,
        reg_pts=line_points,
        names=model.names,
        #draw_tracks=True,
        line_thickness=2,
    )

    url="http://192.168.1.107:8080/shot.jpg"

   
    window_name="Tracker"
    #redimensionnement
    cv2.namedWindow(window_name,cv2.WINDOW_NORMAL)
    if fullscreen:
        cv2.setWindowProperty(window_name,cv2.WND_PROP_FULLSCREEN,cv2.WINDOW_FULLSCREEN)
    else:
        #dimension réduite
        cv2.resizeWindow(window_name,800,600)    

    url="http://192.168.1.107:8080/shot.jpg"

    while True:
        img_resp=requests.get(url)
        img_arr=np.array(bytearray(img_resp.content),dtype=np.uint8)
        img=cv2.imdecode(img_arr,-1)
        img=imutils.resize(img,width=800,height=600)
        ###cv2.imshow("Jojo Phone",img)

        if cv2.waitKey(1)==ord('q'):
            break
        tracks=model.track(img,persist=True,show=False)
        img=counter.start_counting(img,tracks)
        #video_writer.write(img)
        cv2.imshow("Tracker",img)
        if cv2.waitKey(1)==ord('q'):
            break
        if cv2.waitKey(1)==ord('p'):
            fullscreen=True
        elif cv2.waitKey(1)==ord('r'):   
            fullscreen=False

    
    #cap.release()
    #video_writer.release()
    cv2.destroyAllWindows()


count_objects_in_region("yolov8s.pt")







