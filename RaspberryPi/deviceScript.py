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

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #setting max width, height, & fps, raspberrpy pi: 640x480, 1280x720, 1280x960, 1920x1080
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, 30)

width  = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Camera initialized at {int(width)}x{int(height)} @ {int(fps)} FPS")

vehicle_zones = {
    'carsL1_West':  (20, 440, 245, 480),
    'carsL2_East':  (245, 440, 395, 480)
}

pedestrian_zones = {
    'pedsW1_West':  (20, 35, 320, 410), #no touch
    'pedsW2_East':  (395, 440, 640, 480),
    'carsL3_North': (320, 35, 640, 410) #no touch
}

selected_zone = None
dragging = False
drag_offset = (0, 0)
zone_type = None  # 'vehicle' or 'pedestrian'

def point_in_rect(x, y, rect):
    x1, y1, x2, y2 = rect
    return x1 <= x <= x2 and y1 <= y <= y2

def mouse_callback(event, x, y, flags, param):
    global selected_zone, dragging, drag_offset, zone_type
    all_zones = {**vehicle_zones, **pedestrian_zones}
    if event == cv2.EVENT_LBUTTONDOWN:
        for name, rect in all_zones.items():
            if point_in_rect(x, y, rect):
                selected_zone = name
                zone_type = 'vehicle' if name.startswith('cars') else 'pedestrian'
                rx1, ry1, rx2, ry2 = rect
                drag_offset = (x - rx1, y - ry1)
                dragging = True
                break
    elif event == cv2.EVENT_MOUSEMOVE and dragging and selected_zone:
        # Move the rectangle (top-left corner follows mouse, keep size)
        all_zones = vehicle_zones if zone_type == 'vehicle' else pedestrian_zones
        rx1, ry1, rx2, ry2 = all_zones[selected_zone]
        w, h = rx2 - rx1, ry2 - ry1
        new_x1 = x - drag_offset[0]
        new_y1 = y - drag_offset[1]
        all_zones[selected_zone] = (new_x1, new_y1, new_x1 + w, new_y1 + h)
    elif event == cv2.EVENT_LBUTTONUP:
        dragging = False
        selected_zone = None

cv2.namedWindow('YOLOv8 Zone Detection')
cv2.setMouseCallback('YOLOv8 Zone Detection', mouse_callback)

if not cap.isOpened():
    print("Camera not detected.")
    exit()

frame_count = 0
desired_classes = [0, 2, 3, 5, 7] # person: 0, car: 2, motorcycle: 3, bus: 5, truck: 7
TARGET_WIDTH = 640
TARGET_HEIGHT = 480
last_results = None
waitTimeL1_L2, waitTimeL3, waitTimeW1_W2 = 0, 0, 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 1. Resize FRAME immediately. 
    # This ensures drawing coordinates and YOLO coordinates always match.
    frame = cv2.resize(frame, (TARGET_WIDTH, TARGET_HEIGHT))

    # 2. ---- Run YOLO only every 4 frames ----
    if frame_count % 4 == 0:
        # We pass the already resized frame
        results = model(frame, imgsz=TARGET_WIDTH, classes=desired_classes, verbose=True)
        if results:
            last_results = results[0]
            
        # OPTIONAL: Update DB only when we have new data to save bandwidth/CPU
        # Moving this here reduces lag significantly
        data = {
            'carsL1_West': zone_counts['carsL1_West'] if 'zone_counts' in locals() else 0,
            'carsL2_East': zone_counts['carsL2_East'] if 'zone_counts' in locals() else 0,
            'carsL3_North': zone_counts['carsL3_North'] if 'zone_counts' in locals() else 0,
            'pedsW1_West': zone_counts['pedsW1_West'] if 'zone_counts' in locals() else 0,
            'pedsW2_East': zone_counts['pedsW2_East'] if 'zone_counts' in locals() else 0,
            'pedsW3_South': 0,
            'waitTimeL1_L2': waitTimeL1_L2,
            'waitTimeL3': waitTimeL3,
            'waitTimeW1_W2': waitTimeW1_W2
        }
        # Use update instead of set to reduce overhead if possible, or run this in a separate thread
        db.reference('trafficData').set(data)
    # --------------------------------------

    # If we haven't had a successful detection yet, skip drawing
    if last_results is None:
        frame_count += 1
        continue

    # 3. ---- Reset counts for THIS frame ----
    zone_counts = {
        'carsL1_West': 0, 'carsL2_East': 0, 'carsL3_North': 0,
        'pedsW1_West': 0, 'pedsW2_East': 0
    }

    # 4. ---- Iterate over CACHED results ----
    # Even if we didn't run YOLO this frame, we process the old boxes
    # so the counters and visuals persist smoothly.
    if last_results.boxes:
        for box, cls in zip(last_results.boxes.xyxy, last_results.boxes.cls):
            x1, y1, x2, y2 = map(int, box)
            class_id = int(cls)
            cx = (x1 + x2) // 2
            cy = (y1 + y2) // 2

            # Draw bounding box (Optional: Visual confirmation)
            # cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 0), 1)
            
            # Draw center point
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)

            # Vehicle zones
            if class_id in [2, 3, 5, 7]:
                for name, (zx1, zy1, zx2, zy2) in vehicle_zones.items():
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        zone_counts[name] += 1
                        # break # Remove break if a car can be in two overlapping zones

            # Pedestrian zones
            elif class_id == 0:
                for name, (zx1, zy1, zx2, zy2) in pedestrian_zones.items():
                    if zx1 <= cx <= zx2 and zy1 <= cy <= zy2:
                        zone_counts[name] += 1
                        # break

    # 5. ---- Draw Zones and HUD ----
    for name, (x1, y1, x2, y2) in {**vehicle_zones, **pedestrian_zones}.items():
        # Check if zone has activity for color change
        count = zone_counts.get(name, 0)
        color = (0, 255, 0) # Green standard
        if count > 0:
            color = (0, 0, 255) # Red if occupied
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, f"{name}: {count}", (x1 + 5, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    # 6. ---- Accumulate Wait Times ----
    # (Logic remains the same, running every frame)
    carsL1_West = zone_counts['carsL1_West']
    carsL2_East = zone_counts['carsL2_East']
    carsL3_North = zone_counts['carsL3_North']
    pedsW1_West = zone_counts['pedsW1_West']
    pedsW2_East = zone_counts['pedsW2_East']

    waitTimeL1_L2 += carsL1_West + carsL2_East
    waitTimeL3 += carsL3_North
    waitTimeW1_W2 += pedsW1_West + pedsW2_East

    # Reset logic
    if carsL1_West == 0 and carsL2_East == 0:
        waitTimeL1_L2 = 0
    if carsL3_North == 0:
        waitTimeL3 = 0
    if pedsW1_West == 0 and pedsW2_East == 0:
        waitTimeW1_W2 = 0

    # Display stats
    # cv2.putText(frame, f"Wait L1/L2: {waitTimeL1_L2}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)
    # cv2.putText(frame, f"Wait L3: {waitTimeL3}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

    # cv2.putText(frame, f"L1/L2 Wait: {waitTimeL1_L2}s", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    cv2.putText(frame, f"L3 Wait: {waitTimeL3}s", (325, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    cv2.putText(frame, f"W1/W2 Wait: {waitTimeW1_W2}s", (25, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,255,255),2)
    
    cv2.imshow('YOLOv8 Optimized', frame)

    frame_count += 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

#source yolo-env/bin/activate / for windows: .\yolo-env\Scripts\activate
#deactivate