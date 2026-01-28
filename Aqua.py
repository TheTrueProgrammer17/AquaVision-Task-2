import cv2 as cv
import numpy as np

# =========================
# Assigned Target Color
# =========================
ASSIGNED_COLOR = "green"   # Change this to "red", "green", or "yellow" as needed

# =========================
# HSV Color Ranges
# =========================
COLOR_RANGES = {
   # "white": [(np.array([0,0,168]),np.array([179,255,255]))],
   # "blue":   [(np.array([90, 80, 40]), np.array([140, 255, 255]))],
    "red":    [(np.array([0, 80, 40]), np.array([10, 255, 255])),
               (np.array([165, 80, 40]), np.array([180, 255, 255]))],
    "green":  [(np.array([35, 60, 40]), np.array([85, 255, 255]))],
    # "yellow": [(np.array([15, 80, 40]), np.array([40, 255, 255]))]
}


# =========================
# Distance Calibration
# =========================
KNOWN_DISTANCE = 150.0      # cm (distance at which you calibrate)
REAL_GATE_WIDTH = 100.0     # cm (1 meter gate width)
FOCAL_LENGTH = None         # will be calculated once

# =========================
# Camera Setup
# =========================
cap = cv.VideoCapture(2, cv.CAP_V4L2)
cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

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
    if not ret:
        break

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

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    gray = cv.GaussianBlur(gray, (5, 5), 0)

    h, w, _ = frame.shape
    frame_cx, frame_cy = w // 2, h // 2

    detections = {}   # store all detected gates
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # -------------------------
    # Detect all colors
    # -------------------------
    for color_name, ranges in COLOR_RANGES.items():
        mask = None
        for lower, upper in ranges:
            temp_mask = cv.inRange(hsv, lower, upper)
            mask = temp_mask if mask is None else cv.bitwise_or(mask, temp_mask)

        edges = cv.Canny(gray, 50, 150)

        # Morphological cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_DILATE, kernel)
        mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)


        combined_mask = cv.bitwise_or(combined_mask, mask)

        # Find contours for this color
        contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
        if contours:
            # 1. Filter out tiny noise immediately
            valid_contours = [c for c in contours if cv.contourArea(c) > 400]
            
            if valid_contours:
                # 2. Grab the largest potential gate
                gate = max(valid_contours, key=cv.contourArea)
                
                # 3. Shape Approximation: Simplifies the shape to see if it's actually rectangular
                peri = cv.arcLength(gate, True)
                approx = cv.approxPolyDP(gate, 0.04 * peri, True)
                
                # 4. Only proceed if it looks like a polygon with 4-8 corners
                if 4 <= len(approx) <= 8:
                    x, y, bw, bh = cv.boundingRect(gate)
                    
                    # 5. Temporal Smoothing
                    if color_name in detections:
                        old_cx, old_cy = detections[color_name]["cx"], detections[color_name]["cy"]
                        cx = int(old_cx * 0.7 + (x + bw // 2) * 0.3)
                        cy = int(old_cy * 0.7 + (y + bh // 2) * 0.3)
                    else:
                        cx, cy = x + bw // 2, y + bh // 2

                    # --- ADD THESE DRAWING LINES BACK ---
                    cv.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
                    cv.putText(frame, f"{color_name}", (x, y - 10),
                               cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    # ------------------------------------

                    # Store the smoothed result
                    detections[color_name] = {"cx": cx, "cy": cy, "bw": bw, "bh": bh, "confidence": 1.0}

    # -------------------------
    # Navigation logic for assigned color
    # -------------------------
    command = "SEARCH"
    distance_cm = None

    # Gate prioritization
    preferred_order = ["green", "red"]
    ASSIGNED_COLOR = None
    for color in preferred_order:
        if color in detections:
            ASSIGNED_COLOR = color
            break

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