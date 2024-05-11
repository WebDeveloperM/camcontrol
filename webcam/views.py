from django.shortcuts import render
from django.http import StreamingHttpResponse
import math
import cv2
import cvzone
from ultralytics import YOLO


# Create your views here.
def index(request):
    return render(request,'index.html')

def stream():
    cap = cv2.VideoCapture(0)

    # cap = cv2.VideoCapture(0)
    # address = "http://192.168.1.3:8080/video"
    # cap.open(address)

    model = YOLO("../utils/ppe.pt")
    # model = YOLO("../utils/yolov5s.pt")
    classNames = ['Hardhat', 'Mask' 'No-Hardhat', 'No-Safety Vest', 'Person', 'Safety Cone', 'Safety Vest']
    myColor = (0, 0, 255)

    while True:
        success, frame = cap.read()
        results = model(frame, stream=True)
        if not success:
            print("Error: failed to capture image")
            break
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                conf = math.ceil((box.conf[0]*100))/100
                cls = int(box.cls[0])
                currentClass = classNames[cls]

                if currentClass == 'Hardhat' or currentClass == 'Mask' or currentClass == 'Safety Vest':
                    myColor = (0,255,0)
                else:
                    myColor = (0,0,255)
                

                cvzone.putTextRect(frame, f"{classNames[cls]} {conf, 1}", (max(0,x1),max(40, y1)), scale=1, thickness=1,
                                colorB = myColor, colorT=(255,255,255), colorR=myColor, offset = 5)

                cv2.rectangle(frame, (x1, y1), (x2, y2), myColor, 3)


        image_bytes = cv2.imencode('.jpg', frame)[1].tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + image_bytes + b'\r\n')  

def video_feed(request):
    return StreamingHttpResponse(stream(), content_type='multipart/x-mixed-replace; boundary=frame')    