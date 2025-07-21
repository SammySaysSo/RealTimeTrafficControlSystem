import cv2
import firebase_admin
from firebase_admin import credentials, db
from ultralytics import YOLO

cred = credentials.Certificate("firebase-creds.json")
firebase_admin.initialize_app(cred, {'databaseURL': 'https://fir-mywebsite-default-rtdb.firebaseio.com/'})

model = YOLO("yolov8n.pt") #not that big for the raspberry pi 4 model B, yolov8n.pt for raspberry pi, yolov8m.pt or yolov8l.pt for windows
cap = cv2.VideoCapture(0) #assuming only one camera, windows: cap = cv2.VideoCapture(0), raspberry pi: cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #so not still framing
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280) #setting max width, height, & fps, raspberrpy pi: 640x480, 1280x720, 1280x960, 1920x1080
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera initialized at {int(width)}x{int(height)} @ {int(fps)} FPS")

vehicle_zones = { # Zone definitions: (x1, y1, x2, y2)
    'carsL1_West':  (50, 400, 500, 550),
    'carsL2_East':  (700, 225, 1250, 375),
    'carsL3_North': (500, 600, 750, 750)
}

pedestrian_zones = {
    'pedsW1_West':  (100, 50, 600, 200),
    'pedsW2_East':  (650, 50, 1150, 200)
}

if not cap.isOpened():
    print("Camera not detected.")
    exit()

frame_count = 0
desired_classes = [0, 2, 3, 5, 7] # person: 0, car: 2, motorcycle: 3, bus: 5, truck: 7
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Uncomment the following lines if you want to resize the frame for performance on raspberry pi
    # if frame_count % 4 == 0:  #run YOLO every 4th frame ~ bc it's very slow since on CPU and not on GPU bc its a raspberry pi ~ still good enough hopefully :(
    #     resized_frame = cv2.resize(frame, (320, 192)) #if need to resize for performance
    #     results = model(resized_frame, imgsz=320, classes=desired_classes)[0]
    #     last_annotated = results.plot()
    #     cv2.imshow('YOLOv8 Detection', last_annotated)
    # elif last_annotated is not None:
    #     cv2.imshow('YOLOv8 Detection', last_annotated)
    # else:
    #     cv2.imshow('YOLOv8 Detection', frame)
    # frame_count += 1

    results = model(frame, imgsz=1280, classes=desired_classes)[0]

    zone_counts = {'carsL1_West': 0, 'carsL2_East': 0, 'carsL3_North': 0,'pedsW1_West': 0, 'pedsW2_East': 0}

    for box, cls in zip(results.boxes.xyxy, results.boxes.cls):
        x1, y1, x2, y2 = map(int, box)
        class_id = int(cls)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # center point

        # Draw detection center
        cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

        # Vehicle zones
        if class_id in [2, 3, 5, 7]:
            for name, (zx1, zy1, zx2, zy2) in vehicle_zones.items():
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_counts[name] += 1
                    break  # one zone per detection

        # Pedestrian zones
        elif class_id == 0:
            for name, (zx1, zy1, zx2, zy2) in pedestrian_zones.items():
                if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                    zone_counts[name] += 1
                    break

    # Draw all zones with labels
    for name, (x1, y1, x2, y2) in {**vehicle_zones, **pedestrian_zones}.items():
        color = (255, 0, 0) if name.startswith('cars') else (0, 255, 0)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        count = zone_counts[name]
        cv2.putText(frame, f"{name}: {count}", (x1 + 5, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # Assign counts to variables
    carsL1_West = zone_counts['carsL1_West']
    carsL2_East = zone_counts['carsL2_East']
    carsL3_North = zone_counts['carsL3_North']
    pedsW1_West = zone_counts['pedsW1_West']
    pedsW2_East = zone_counts['pedsW2_East']

    # Example wait time logic
    waitTimeL1_L2 = (carsL1_West + carsL2_East) * 3  # seconds per vehicle
    waitTimeL3 = carsL3_North * 2
    waitTimeW1_W2 = (pedsW1_West + pedsW2_East) * 4

    data = {
        'carsL1_West': carsL1_West,
        'carsL2_East': carsL2_East,
        'carsL3_North': carsL3_North,
        'pedsW1_West': pedsW1_West,
        'pedsW2_East': pedsW2_East,
        'pedsW3_South': 0,
        'waitTimeL1_L2': waitTimeL1_L2,
        'waitTimeL3': waitTimeL3,
        'waitTimeW1_W2': waitTimeW1_W2
    }
    db.reference('trafficData').set(data)

    # Display wait times
    cv2.putText(frame, f"waitTimeL1_L2: {waitTimeL1_L2}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"waitTimeL3: {waitTimeL3}s", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(frame, f"waitTimeW1_W2: {waitTimeW1_W2}s", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    cv2.imshow('YOLOv8 Zone Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#source yolo-env/bin/activate / for windows: .\yolo-env\Scripts\activate
#deactivate