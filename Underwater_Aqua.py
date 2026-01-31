import cv2 as cv
import numpy as np

def mouse_callback(event, x, y, flags, param):
    if event == cv.EVENT_LBUTTONDOWN and param is not None:
        hsv_val = param[y, x]
        print(f"HSV at ({x}, {y}): {hsv_val}")

cap = cv.VideoCapture(0, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

# =========================
# Assigned Target Color
# =========================
ASSIGNED_COLOR = "green"   # Change this to "red", "green", or "yellow" as needed

# =========================
# HSV Color Ranges
# =========================

# These HSV values have been set on the basis of dataset
COLOR_RANGES = {
    "green": [
        (np.array([70, 120, 100]), np.array([85, 255, 255]))
    ],
    "red": [
        (np.array([0, 30, 50]), np.array([10, 255, 255])),
        (np.array([160, 30, 50]), np.array([180, 255, 255]))
    ]
}

# These values are for testing on land
# COLOR_RANGES = {
   # "white": [(np.array([0,0,168]),np.array([179,255,255]))],
   # "blue":   [(np.array([90, 80, 40]), np.array([140, 255, 255]))],
    # "red":    [(np.array([0, 80, 40]), np.array([10, 255, 255])),
    #            (np.array([165, 80, 40]), np.array([180, 255, 255]))],
    # "green":  [(np.array([35, 60, 40]), np.array([85, 255, 255]))],
    # "yellow": [(np.array([15, 80, 40]), np.array([40, 255, 255]))]
# }

# =========================
# Distance Calibration
# =========================
KNOWN_DISTANCE = 150.0      # cm (distance at which you calibrate)
REAL_GATE_WIDTH = 100.0     # cm (1 meter gate width)
FOCAL_LENGTH = None         # will be calculated once

# =========================
# Camera Setup
# =========================
# frame = cv.imread("/home/shashwat/Documents/AUV/DataSetOfImages/images_new(1)/images_new/GX010683_window1_second68.png") 
# if frame is None: 
#     print("Error: Could not load image.") 
#     exit()

cv.namedWindow("Frame", cv.WINDOW_NORMAL)

# cv.namedWindow("Mask", cv.WINDOW_NORMAL)
# cv.namedWindow("Edges", cv.WINDOW_NORMAL)
cv.resizeWindow("Frame", 800, 600)
# cv.resizeWindow("Mask", 400, 300)
# cv.resizeWindow("Edges", 400, 300)

distance_buffer = []

prev_gate_pos = None

while True:
    ret, frame = cap.read()
    if not ret or frame is None:
        continue


    # -------------------------
    # Preprocessing
    # -------------------------
    # frame = cv.bilateralFilter(frame, d=9, sigmaColor=75, sigmaSpace=75)

    # White balance / contrast enhancement
    lab = cv.cvtColor(frame, cv.COLOR_BGR2LAB)
    l, a, b = cv.split(lab)
    clahe = cv.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    lab = cv.merge((l,a,b))
    frame = cv.cvtColor(lab, cv.COLOR_LAB2BGR) 

    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
    h, s, v = cv.split(hsv)
    # v = cv.equalizeHist(v)
    clahe_v = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    v = clahe_v.apply(v)
    hsv = cv.merge((h, s, v))
    cv.setMouseCallback("Frame", mouse_callback, hsv)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    h, w, _ = frame.shape
    frame_cx, frame_cy = w // 2, h // 2

    detections = {}   # store all detected gates
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # -------------------------
    # Detect all colors
    # -------------------------
    # -------------------------
    # Detect all colors
    # -------------------------
    # -------------------------
    # Detect all colors
    # -------------------------
    for color_name, ranges in COLOR_RANGES.items():
        if color_name == "red":
            # HSV mask for red
            mask = None
            for lower, upper in ranges:
                temp_mask = cv.inRange(hsv, lower, upper)
                mask = temp_mask if mask is None else cv.bitwise_or(mask, temp_mask)

            # Channel dominance mask
            b, g, r = cv.split(frame)
            mask_dom = (r > 80) & (r > g + 30) & (r > b + 30)
            mask_dom = mask_dom.astype(np.uint8) * 255

            # Combine HSV + dominance
            mask = cv.bitwise_or(mask, mask_dom)

        else:  # green
            mask = None
            for lower, upper in ranges:
                temp_mask = cv.inRange(hsv, lower, upper)
                mask = temp_mask if mask is None else cv.bitwise_or(mask, temp_mask)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)

        cv.imshow(f"{color_name} Mask", mask)

        combined_mask = cv.bitwise_or(combined_mask, mask)

        # Find contours
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            valid_contours = [c for c in contours if cv.contourArea(c) > 200]
            if valid_contours:
                if len(valid_contours) > 1:
                    merged = np.vstack(valid_contours)
                    gate = cv.convexHull(merged)
                else:
                    gate = valid_contours[0]

                x, y, bw, bh = cv.boundingRect(gate)

                peri = cv.arcLength(gate, True)
                approx = cv.approxPolyDP(gate, 0.04 * peri, True)
                x, y, bw, bh = cv.boundingRect(gate)

                if bw > w * 0.9 or bh > h * 0.9:
                    continue  # skip huge blobs like water

                if len(approx) >= 4 and cv.contourArea(gate) > 150:
                    cx, cy = x + bw // 2, y + bh // 2

                    # Add smoothing here
                    if prev_gate_pos is not None:
                        cx = int(prev_gate_pos[0] * 0.7 + cx * 0.3)
                        cy = int(prev_gate_pos[1] * 0.7 + cy * 0.3)
                    prev_gate_pos = (cx, cy)

                    cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv.putText(frame, f"{color_name}", (x, y - 10),
                            cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    detections[color_name] = {"cx": cx, "cy": cy, "bw": bw, "bh": bh, "confidence": 1.0}


    # -------------------------
    # Navigation logic for assigned color
    # -------------------------
    command = "SEARCH"
    distance_cm = None

    # Gate prioritization
    ASSIGNED_COLOR = None
    target = None

    # If multiple green gates detected, pick the nearest (largest bounding box)
    green_candidates = [d for c, d in detections.items() if c == "green"]
    if green_candidates:
        nearest = max(green_candidates, key=lambda d: d["bw"] + d["bh"])
        target = nearest
        ASSIGNED_COLOR = "green"
    elif "red" in detections:
        target = detections["red"]
        ASSIGNED_COLOR = "red"

    # Message logic — only show if a gate is detected
    if "green" in detections and "red" in detections:
        cv.putText(frame, "Prioritizing GREEN gate over RED", (10, 120),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif ASSIGNED_COLOR == "green":
        cv.putText(frame, "Going towards GREEN gate", (10, 120),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    elif ASSIGNED_COLOR == "red":
        cv.putText(frame, "Going towards RED gate", (10, 120),
                cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)



    if ASSIGNED_COLOR in detections:
        target = detections[ASSIGNED_COLOR]
        gate_cx, gate_cy = target["cx"], target["cy"]

        cv.circle(frame, (gate_cx, gate_cy), 5, (0, 0, 255), -1)

        # Distance estimation
        # Distance estimation using average gate size
        bw = target["bw"]
        bh = target["bh"]
        avg_size = (bw + bh) / 2

        if FOCAL_LENGTH is None:
            FOCAL_LENGTH = (avg_size * KNOWN_DISTANCE) / REAL_GATE_WIDTH

        distance_cm = (REAL_GATE_WIDTH * FOCAL_LENGTH) / avg_size

        # Smooth the distance values
        distance_buffer.append(distance_cm)
        if len(distance_buffer) > 5:
            distance_buffer.pop(0)
        smoothed_distance = sum(distance_buffer) / len(distance_buffer)

        cv.putText(frame, f"Distance: {int(smoothed_distance)} cm", (10, 60),
                cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        # Alignment logic
        dx = gate_cx - frame_cx   # horizontal offset (pixels)
        dy = gate_cy - frame_cy   # vertical offset (pixels)

        # Offset calculator (normalized to frame size)
        offset_x = dx / (w / 2)   # -1.0 (far left) to +1.0 (far right)
        offset_y = -dy / (h / 2)   # +1.0 (top) to -1.0 (bottom)

        cv.putText(frame, f"Offset X: {offset_x:.2f}", (10, 150),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv.putText(frame, f"Offset Y: {offset_y:.2f}", (10, 180),
                    cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Command logic using offsets
        if abs(dx) > abs(dy):
            if dx > 40:
                command = "MOVE RIGHT"
            elif dx < -40:
                command = "MOVE LEFT"
            else:
                command = "CENTERED"
        else:
            if dy > 30:
                command = "MOVE UP"
            elif dy < -30:
                command = "MOVE DOWN"
            else:
                command = "CENTERED"


    # -------------------------
    # Display
    # -------------------------
    cv.line(frame, (frame_cx, 0), (frame_cx, h), (255, 255, 255), 1)
    cv.line(frame, (0, frame_cy), (w, frame_cy), (255, 255, 255), 1)

    cv.putText(frame, f"COMMAND: {command}", (10, 30),
            cv.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # Apply blue tint overlay
    # blue_overlay = np.full(frame.shape, (255, 0, 0), dtype=np.uint8)  # BGR blue
    # alpha = 0.3  # transparency factor
    # frame = cv.addWeighted(blue_overlay, alpha, frame, 1 - alpha, 0)

    cv.imshow("Frame", frame)

    # cv.imshow("Mask", combined_mask)
    # cv.imshow("Edges", edges)

    if cv.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
