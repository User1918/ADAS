# -*- coding: utf-8 -*-
import numpy as np
import cv2
import time
import ncnn # Import the ncnn Python binding
from picamera2 import Picamera2 # Import Picamera2 library
import serial
import struct # Needed for packing binary data
import re # Import regular expressions for parsing speed limits

# --- Configuration ---
# NCNN Traffic Sign Detection Config
PARAM_PATH = "/home/labthayphuc/Desktop/yolov11ncnn/model.ncnn.param" # <<< UPDATE THIS PATH
BIN_PATH = "/home/labthayphuc/Desktop/yolov11ncnn/model.ncnn.bin"     # <<< UPDATE THIS PATH
NCNN_INPUT_SIZE = 640           # Input size for NCNN model (MUST BE 640 for modified preprocessing)
NCNN_CONFIDENCE_THRESHOLD = 0.35 # Min confidence for NCNN detection (Adjust if needed)
NCNN_NMS_THRESHOLD = 0.45       # NMS threshold for NCNN detection
NCNN_NUM_CLASSES = 17           # Number of classes for NCNN model
NCNN_CLASS_NAMES = [            # Class names for NCNN model
    "Pedestrian Crossing", "Radar", "Speed Limit -100-", "Speed Limit -120-",
    "Speed Limit -20-", "Speed Limit -30-", "Speed Limit -40-", "Speed Limit -50-",
    "Speed Limit -60-", "Speed Limit -70-", "Speed Limit -80-", "Speed Limit -90-",
    "Stop Sign", "Traffic Light -Green-", "Traffic Light -Off-",
    "Traffic Light -Red-", "Traffic Light -Yellow-"
]
NCNN_INPUT_NAME = "in0"         # Input layer name for NCNN
NCNN_OUTPUT_NAME = "out0"       # Output layer name for NCNN
NCNN_MEAN_VALS = [0.0, 0.0, 0.0] # Mean values for NCNN normalization
NCNN_NORM_VALS = [1/255.0, 1/255.0, 1/255.0] # Norm values for NCNN normalization

# Lane Keeping & Motor Control Config
LANE_ROI_CROP = (100, 380, 540, 480) # Example: (x1, y1, x2, y2) -> Within 640x480 warped image
LANE_RESIZE_DIM = (640, 480)         # resize_dim for warpImg
SERIAL_PORT = '/dev/ttyUSB0'         # Check if this is correct using 'ls /dev/ttyUSB*'
BAUD_RATE = 115200
PICAM_SIZE = (640, 480)              # Input camera resolution
PICAM_FRAMERATE = 60                 # Target framerate (actual FPS depends on processing)

# Speed Control Config
DEFAULT_MAX_SPEED = 40          # Default speed limit if no sign detected
MANUAL_MAX_SPEED = 50           # Absolute max speed achievable with 'w' key
MANUAL_ACCELERATION = 5
AUTO_MODE_SPEED_STRAIGHT = 30   # Speed for 'a' when straight
AUTO_MODE_SPEED_CURVE = 20      # Speed for 'a' when turning


# --- Global Variables ---
canny_edge = None; frame = None; gray = None; imgWarp = None; lines = None
roi = None; roi_haha = None; points = None
left_point = -1; right_point = -1; interested_line_y = 0; im_height = 0; im_width = 0
lane_width = 500; lane_width_max = 0; cte_f = 0.0
speed_motor_requested = 0; last_speed_sent = 0; current_speed_limit = DEFAULT_MAX_SPEED
flag = 1; steering = 90
net = None # Initialize net globally
ser = None # Initialize ser globally

# --- Initializations ---

# Serial bus setup
print(f"Initializing Serial on {SERIAL_PORT} at {BAUD_RATE} baud...")
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1) # Short timeout
    time.sleep(2) # Give serial port time to initialize
    if ser.is_open:
        print("Serial port initialized successfully.")
    else:
        print("CRITICAL: Serial port could not be opened.")
        exit()
except serial.SerialException as e:
    print(f"CRITICAL: Serial Error opening {SERIAL_PORT}: {e}")
    print("Check if the port exists, if you have permissions (dialout group), and if it's used by another program.")
    exit()
except Exception as e:
    print(f"CRITICAL: Unexpected error opening serial port: {e}")
    exit()

# NCNN Initialization with Vulkan Check
print("Initializing NCNN Net...")
net = ncnn.Net() # Create the Net object

print("Checking for Vulkan GPU...")
gpu_count = 0
try:
    gpu_count = ncnn.get_gpu_count()
except Exception as e:
    print(f"Warning: Could not query GPU count - {e}. Assuming CPU only.")

if gpu_count > 0: # Check if GPUs are found (corrected logic)
    print(f"Vulkan GPU detected ({gpu_count} device(s)). Enabling Vulkan acceleration.")
    net.opt.use_vulkan_compute = True
else:
    print("Vulkan GPU not found or check failed. Using CPU threads.")
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4 # Use 4 threads for Pi 5 CPU (adjust if needed)

# Load model parameters and weights *after* setting options
print("Loading NCNN model parameters and weights...")
try:
    start_load_time = time.time()
    if net.load_param(PARAM_PATH) != 0:
        print(f"CRITICAL: Load Param Error: {PARAM_PATH}")
        exit()
    if net.load_model(BIN_PATH) != 0:
        print(f"CRITICAL: Load Model Error: {BIN_PATH}")
        exit()
    end_load_time = time.time()
    print(f"NCNN model loaded successfully in {end_load_time - start_load_time:.4f} seconds.")
except Exception as e:
    print(f"CRITICAL: Exception during NCNN model loading: {e}")
    exit()

# PiCamera2 setup
print("Initializing Picamera2...")
piCam = Picamera2()
try:
    config = piCam.create_preview_configuration(main={"size": PICAM_SIZE, "format": "RGB888"})
    piCam.configure(config)
    # Try setting framerate via controls *after* configuration
    piCam.set_controls({"FrameRate": PICAM_FRAMERATE})
    piCam.start()
    time.sleep(1.0) # Allow camera to settle
    print("Picamera2 started.")
except Exception as e:
    print(f"CRITICAL: Picamera2 Error: {e}")
    if ser and ser.is_open: ser.close() # Clean up serial if camera fails
    exit()

# --- Helper Functions ---
# (Keep all helper functions: picam, getPoints, warpImg, drawPoints,
# SteeringAngle, handle_key_input, PID, parse_speed_limit,
# detect_signs_and_get_results - exactly as before, ensuring
# detect_signs_and_get_results uses the global 'net')
def picam():
    global frame, gray, canny_edge
    try:
        # Capture frame using capture_array()
        rgb_frame = piCam.capture_array()
        if rgb_frame is None:
            frame = None; gray = None; canny_edge = None; return

        # Convert RGB to BGR for OpenCV convention
        frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply Gaussian Blur
        blur = cv2.GaussianBlur(gray, (11, 11), 0) # Kernel size might need tuning

        # Apply Canny Edge Detection
        thresh_low = 100  # Lower threshold for Canny (tune if needed)
        thresh_high = 175 # Upper threshold for Canny (tune if needed)
        canny_edge = cv2.Canny(blur, thresh_low, thresh_high, apertureSize=3)

    except Exception as e:
        print(f"Error in picam function: {e}")
        frame = None; gray = None; canny_edge = None

def getPoints(wT=640, hT=480):
    global points
    # These points define the region in the *original* camera view
    # that corresponds to a rectangular region in the warped top-down view.
    # Adjust these based on your camera mounting angle and height.
    # Values are relative to the image dimensions (wT, hT).
    widthTop = 70      # How far from the side edges the top corners of the trapezoid are
    heightTop = 95     # How far down from the top edge the top corners are
    widthBottom = -750 # How far *outside* the bottom edges the bottom corners are (creates wider base)
    heightBottom = 380 # How far down from the top edge the bottom corners are

    points = np.float32([
        (widthTop, heightTop),                 # Top-Left
        (wT - widthTop, heightTop),            # Top-Right
        (widthBottom, heightBottom),           # Bottom-Left
        (wT - widthBottom, heightBottom)       # Bottom-Right
    ])

def warpImg(roi_crop=LANE_ROI_CROP, resize_dim=LANE_RESIZE_DIM):
    global imgWarp, lines, roi, roi_haha
    if canny_edge is None:
        imgWarp = None; roi = None; roi_haha = None; lines = None
        return None

    h, w = canny_edge.shape[:2]

    # Ensure points are defined for the current image dimensions
    if points is None or len(points) != 4:
        getPoints(w, h) # Define points based on actual image size

    pts1 = np.float32(points)
    # Define the target rectangle in the output warped image
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    try:
        # Calculate the perspective transform matrix
        matrix = cv2.getPerspectiveTransform(pts1, pts2)
        # Apply the perspective warp to the Canny edge image
        imgWarp = cv2.warpPerspective(canny_edge, matrix, (w, h))
    except cv2.error as e:
        print(f"Error during perspective warp: {e}")
        imgWarp = None; roi = None; roi_haha = None; lines = None
        return None

    # --- Crop the Region of Interest (ROI) from the WARPED image ---
    x1, y1, x2, y2 = roi_crop

    # Clamp ROI coordinates to be within the warped image bounds
    y1_clamped = max(0, min(y1, h))
    y2_clamped = max(y1_clamped, min(y2, h)) # Ensure y2 >= y1
    x1_clamped = max(0, min(x1, w))
    x2_clamped = max(x1_clamped, min(x2, w)) # Ensure x2 >= x1

    # Check if the clamped ROI is valid
    if y1_clamped >= y2_clamped or x1_clamped >= x2_clamped:
        print(f"ROI Error: Invalid coordinates after clamping. Original: {roi_crop}, Clamped: {(x1_clamped, y1_clamped, x2_clamped, y2_clamped)}")
        roi = None; roi_haha = None; lines = None
        return None

    # Extract the ROI
    roi = imgWarp[y1_clamped:y2_clamped, x1_clamped:x2_clamped]

    # Check if ROI extraction was successful
    if roi is None or roi.size == 0:
        print("ROI is empty after cropping.")
        roi_haha = None; lines = None
        return None

    # Resize the cropped ROI to the desired dimensions for consistent processing
    try:
        roi_haha = cv2.resize(roi, resize_dim, interpolation=cv2.INTER_LINEAR)
    except cv2.error as e:
        print(f"Error resizing ROI: {e}")
        roi_haha = None; lines = None
        return None

    # Apply Hough Line Transform P to detect lines in the resized ROI
    # Adjust parameters as needed for your lighting/lane conditions
    lines = cv2.HoughLinesP(
        roi_haha,
        rho=1,                # Distance resolution of the accumulator in pixels
        theta=np.pi / 180,    # Angle resolution of the accumulator in radians
        threshold=170,        # Accumulator threshold parameter. Only lines receiving enough votes get returned.
        minLineLength=100,    # Minimum line length. Line segments shorter than this are rejected.
        maxLineGap=20         # Maximum allowed gap between points on the same line to link them.
    )

    return roi_haha # Return the resized ROI (roi_haha) for display/further use

def drawPoints():
    global frame, points
    if frame is None or points is None: return
    display_frame = frame.copy()
    color = (0, 0, 255) # Red color for points
    for x in range(4):
        try:
            # Ensure points are integers for drawing
            pt = (int(points[x][0]), int(points[x][1]))
            cv2.circle(display_frame, pt, 10, color, cv2.FILLED)
        except IndexError:
            print("Error accessing points for drawing.")
            pass # Continue if a point is missing
    # cv2.imshow("Perspective Points on Frame", display_frame) # Optional display

def SteeringAngle():
    global left_point, right_point, interested_line_y, im_height, im_width
    global lane_width, lane_width_max, cte_f

    if roi_haha is None or roi_haha.size == 0:
        cte_f = 0.0 # No image to process, assume center
        return

    im_height, im_width = roi_haha.shape[:2]
    if im_height == 0 or im_width == 0:
        cte_f = 0.0 # Invalid image dimensions
        return

    # Reset points for each frame
    left_point = -1
    right_point = -1
    center_img = im_width // 2

    # Define the vertical position (row) in the image where we want to measure the lane center
    # Typically lower down in the warped image (closer to the vehicle)
    # Using 85% down from the top of the resized ROI (roi_haha)
    set_point = max(0, min(int(im_height * 0.85), im_height - 1))
    interested_line_y = set_point

    # Initialize lane_width_max if it hasn't been set
    if lane_width_max <= 0:
        lane_width_max = int(im_width * 0.8) # Assume max lane width is 80% of image width initially

    # Extract the horizontal line (row) of pixels at the interested_line_y
    interested_line = roi_haha[interested_line_y, :]

    # Find white pixels (edges) on the left and right halves of this line
    # Left side
    left_indices = np.where(interested_line[:center_img] > 0)[0] # Find indices of non-zero pixels
    if len(left_indices) > 0:
        left_point = left_indices[-1] # Take the rightmost white pixel found on the left side

    # Right side
    right_indices = np.where(interested_line[center_img:] > 0)[0] # Find indices relative to the center
    if len(right_indices) > 0:
        right_point = center_img + right_indices[0] # Take the leftmost white pixel found on the right side (add center offset back)

    # Estimate missing points based on the known lane width if one side is detected
    if left_point != -1 and right_point == -1:
        # Only left detected, estimate right based on max lane width
        right_point = left_point + lane_width_max
    elif right_point != -1 and left_point == -1:
        # Only right detected, estimate left based on max lane width
        left_point = right_point - lane_width_max
    elif left_point != -1 and right_point != -1:
        # Both detected, update the max lane width estimate (optional, can add smoothing)
        current_lane_width = right_point - left_point
        # Update lane_width_max, ensuring it's reasonable
        lane_width_max = max(current_lane_width, 50) # Min width 50 pixels
        lane_width_max = min(lane_width_max, im_width) # Max width is image width

    # Calculate the lane's center point based on detected/estimated edges
    if left_point != -1 or right_point != -1:
        if left_point == -1: # Only right was found (or estimated)
            mid_point = right_point - lane_width_max / 2
        elif right_point == -1: # Only left was found (or estimated)
            mid_point = left_point + lane_width_max / 2
        else: # Both points available
            mid_point = (right_point + left_point) / 2

        # Calculate Cross Track Error (CTE)
        # CTE = difference between image center and calculated lane center
        cte_f = center_img - mid_point
    else:
        # No lane edges detected on the interested line
        cte_f = 0.0

def handle_key_input(key):
    global speed_motor_requested, flag
    new_speed_req = speed_motor_requested # Start with current speed

    if key == ord('w'): # Increase speed (Manual Forward)
        new_speed_req = min(speed_motor_requested + MANUAL_ACCELERATION, MANUAL_MAX_SPEED)
        flag = 1 # Set direction flag to forward
        print(f"Key 'w': Speed Req -> {new_speed_req}")
    elif key == ord('s'): # Decrease speed (Manual Slowdown/Stop)
        new_speed_req = max(speed_motor_requested - MANUAL_ACCELERATION, 0)
        flag = 0 if new_speed_req == 0 else 1 # Set flag to stop only if speed is 0
        print(f"Key 's': Speed Req -> {new_speed_req}")
    elif key == ord('a'): # Auto speed mode (adjusts based on curve)
        # Determine target speed based on how straight the path is (using CTE)
        target_speed = AUTO_MODE_SPEED_STRAIGHT if abs(cte_f) < 10 else AUTO_MODE_SPEED_CURVE
        new_speed_req = target_speed
        flag = 1 # Auto mode is always forward
        print(f"Key 'a': Auto Speed Req -> {new_speed_req} (CTE: {cte_f:.2f})")
    elif key == ord('x'): # Emergency Stop
        new_speed_req = 0
        flag = 0 # Set direction flag to stop
        print("Key 'x': Emergency Stop Req -> 0")

    # Update global requested speed only if it changed
    if new_speed_req != speed_motor_requested:
        speed_motor_requested = new_speed_req

class PID:
    def __init__(self, kp, ki, kd, integral_limit=500, output_limit=(-60, 60)):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.cte_previous = 0.0
        self.integral = 0.0
        self.integral_limit = integral_limit
        self.output_min, self.output_max = output_limit
        self.last_time = time.time()

    def update(self, cte):
        current_time = time.time()
        delta_time = current_time - self.last_time

        # Avoid division by zero or excessively small dt
        if delta_time <= 1e-6:
            delta_time = 1e-6 # Use a small non-zero value

        # Proportional term
        p_term = self.kp * cte

        # Integral term (with anti-windup)
        self.integral += cte * delta_time
        self.integral = max(min(self.integral, self.integral_limit), -self.integral_limit) # Clamp integral
        i_term = self.ki * self.integral

        # Derivative term (with check for valid delta_time)
        derivative = (cte - self.cte_previous) / delta_time
        d_term = self.kd * derivative

        # Update previous CTE and time for next iteration
        self.cte_previous = cte
        self.last_time = current_time

        # Calculate PID output
        pid_output = p_term + i_term + d_term

        # Clamp PID output to defined limits (e.g., servo deviation range)
        steering_deviation = max(min(pid_output, self.output_max), self.output_min)

        # Convert steering deviation to servo angle (assuming 90 is center)
        # Ensure the final angle is within servo limits (e.g., 0-180)
        steering_servo_angle = round(90 + steering_deviation) # Center + deviation
        steering_servo_angle = max(min(steering_servo_angle, 180), 0) # Clamp to 0-180

        return steering_servo_angle # Return the final clamped servo angle

def parse_speed_limit(class_name):
    # Use regular expression to find numbers within "Speed Limit -XXX-"
    match = re.search(r'Speed Limit -(\d+)-', class_name)
    if match:
        try:
            # Extract the matched digits (group 1) and convert to integer
            return int(match.group(1))
        except ValueError:
            # Handle cases where matched group is not a valid integer (shouldn't happen with \d+)
            return None
    # Return None if the class name doesn't match the speed limit pattern
    return None

def detect_signs_and_get_results(input_frame_bgr):
    """
    Performs NCNN inference for traffic sign detection on a 640x480 BGR frame.
    Assumes NCNN_INPUT_SIZE is 640 and uses padding without scaling.
    Returns a list of detections, each like: {"name": "ClassName", "confidence": 0.XX}
    """
    global net # Ensure we are using the globally loaded network

    detections_results = []
    if net is None:
        print("Error: NCNN net is not initialized in detect_signs_and_get_results.")
        return detections_results
    if input_frame_bgr is None:
        # print("Debug: No input frame provided to detect_signs.") # Less verbose
        return detections_results

    original_height, original_width = input_frame_bgr.shape[:2]

    # --- Preprocessing for YOLOv5/YOLOX Style Models (Pad, No Scale) ---
    # Check if input frame matches expected size (640x480) for this specific non-scaling logic
    if original_width != 640 or original_height != 480 or NCNN_INPUT_SIZE != 640:
         print(f"ERROR: detect_signs (no scale) expects 640x480 input and 640 NCNN size. Got {original_width}x{original_height}")
         return detections_results # Return empty if size mismatch for this padding method

    try:
        # Create a padded image (letterbox/pillarbox) - assumes NCNN_INPUT_SIZE >= original dimensions
        # Background color (e.g., 114, 114, 114 is common for YOLOv5/X)
        padded_img = np.full((NCNN_INPUT_SIZE, NCNN_INPUT_SIZE, 3), 114, dtype=np.uint8)
        # Calculate padding needed
        dw = (NCNN_INPUT_SIZE - original_width) // 2
        dh = (NCNN_INPUT_SIZE - original_height) // 2
        # Place the original image in the center of the padded image
        padded_img[dh:dh+original_height, dw:dw+original_width, :] = input_frame_bgr

        # Convert the padded NumPy array to NCNN's Mat format
        # PIXEL_BGR assumes the input numpy array is in BGR order
        mat_in = ncnn.Mat.from_pixels(padded_img, ncnn.Mat.PixelType.PIXEL_BGR, NCNN_INPUT_SIZE, NCNN_INPUT_SIZE)

        # Apply normalization (subtract mean, divide by norm vals)
        # NCNN_MEAN_VALS = [0.0, 0.0, 0.0]
        # NCNN_NORM_VALS = [1/255.0, 1/255.0, 1/255.0] scales pixels to [0, 1]
        mat_in.substract_mean_normalize(NCNN_MEAN_VALS, NCNN_NORM_VALS)

    except Exception as e:
        print(f"Error during NCNN preprocessing: {e}")
        return detections_results

    # --- NCNN Inference ---
    try:
        # Create an extractor object from the network
        ex = net.create_extractor()
        # Input the preprocessed data into the specified input layer
        ex.input(NCNN_INPUT_NAME, mat_in)
        # Extract the output from the specified output layer
        ret_extract, mat_out = ex.extract(NCNN_OUTPUT_NAME)
        if ret_extract != 0:
             # print(f"Debug: NCNN extraction failed with code {ret_extract}") # Less verbose
             return detections_results # Extraction failed

    except Exception as e:
        print(f"Error during NCNN inference: {e}")
        return detections_results

    # --- Postprocessing ---
    try:
        # Convert the output Mat to a NumPy array
        # Shape might vary based on model output format (e.g., [1, num_detections, 5+num_classes] or [num_detections, 5+num_classes])
        output_data = np.array(mat_out)

        # Handle potential extra dimension (e.g., batch size of 1)
        if len(output_data.shape) == 3 and output_data.shape[0] == 1:
            output_data = output_data[0] # Remove the batch dimension

        # Handle models that might output [5+num_classes, num_detections] - Transpose if needed
        if len(output_data.shape) == 2 and output_data.shape[0] == (NCNN_NUM_CLASSES + 4): # Check if rows match expected fields
             output_data = output_data.T # Transpose to [num_detections, 5+num_classes]

        # Basic sanity check on output shape
        if len(output_data.shape) != 2:
            # print(f"Debug: Unexpected NCNN output shape: {output_data.shape}") # Less verbose
            return detections_results

        num_detections, detection_size = output_data.shape
        expected_size = 4 + NCNN_NUM_CLASSES # cx, cy, w, h + class scores
        # print(f"Debug: Output shape: {output_data.shape}, Expected size: {expected_size}") # Debug print

        if detection_size != expected_size:
            print(f"Debug: Mismatch between detection size ({detection_size}) and expected size ({expected_size}). Check NCNN_NUM_CLASSES.")
            return detections_results

        boxes = []         # Store bounding boxes [x, y, w, h] (relative coordinates initially)
        confidences = []   # Store confidence scores
        class_ids = []     # Store class IDs

        # Iterate through each detection row
        for i in range(num_detections):
            detection = output_data[i]
            class_scores = detection[4:] # Scores start from index 4
            confidence = np.max(class_scores) # Overall confidence is the max score for this box

            # Filter detections by confidence threshold
            if confidence >= NCNN_CONFIDENCE_THRESHOLD:
                class_id = np.argmax(class_scores) # Get the ID of the class with the highest score

                # Note: Bounding box extraction depends on the model's output format
                # Assuming output is [cx, cy, w, h, conf, class_scores...] - NCNN might differ
                # Since NMSBoxes requires boxes, but we don't need the exact coords here,
                # we can just append dummy boxes if only class name and conf are needed after NMS.
                # If you need box coordinates later, extract cx, cy, w, h here and convert.
                boxes.append([0, 0, 1, 1]) # Dummy box for NMS logic if coords aren't needed
                confidences.append(float(confidence))
                class_ids.append(class_id)

        # Apply Non-Maximum Suppression (NMS) if any boxes passed the confidence threshold
        if boxes:
             # Use OpenCV's DNN NMSBoxes function
             # Requires boxes in format [x, y, w, h] (top-left corner + width/height)
             # Since we used dummy boxes, NMS will primarily work based on confidence overlap if coords were real
             indices = cv2.dnn.NMSBoxes(boxes, confidences, NCNN_CONFIDENCE_THRESHOLD, NCNN_NMS_THRESHOLD)

             if len(indices) > 0:
                 # Ensure indices are flattened correctly (can be nested list/array)
                 if isinstance(indices[0], (list, np.ndarray)):
                     indices = indices.flatten()

                 processed_indices = set() # Avoid duplicates if NMS returns overlapping indices somehow
                 for idx in indices:
                     i = int(idx) # Get the original index from NMS result
                     # Check bounds and avoid duplicates
                     if 0 <= i < len(class_ids) and i not in processed_indices:
                         confidence_nms = confidences[i]
                         class_id_nms = class_ids[i]
                         processed_indices.add(i)

                         # Get class name from ID
                         if 0 <= class_id_nms < len(NCNN_CLASS_NAMES):
                             class_name = NCNN_CLASS_NAMES[class_id_nms]
                         else:
                             class_name = f"ID:{class_id_nms}" # Fallback if ID is out of range

                         # Append the final result after NMS
                         detections_results.append({
                             "name": class_name,
                             "confidence": confidence_nms
                             # Add bounding box here if you extracted and converted coordinates
                         })
    except Exception as e:
        print(f"Error during NCNN postprocessing: {e}")
        # Fall through to return potentially empty list

    return detections_results


# --- Main Execution ---
def main():
    global steering, last_speed_sent, speed_motor_requested, flag, current_speed_limit, net, ser # Add net & ser

    if net is None: # Double-check net initialization
        print("CRITICAL: NCNN Net object was not initialized before main loop!")
        if ser and ser.is_open: ser.close()
        if 'piCam' in globals() and piCam.started: piCam.stop()
        return
    if ser is None or not ser.is_open:
        print("CRITICAL: Serial port is not open before main loop!")
        if 'piCam' in globals() and piCam.started: piCam.stop()
        return

    # --- PID Controller Setup ---
    # ** TUNE THESE PID GAINS based on testing **
    kp = 0.4  # Proportional gain (Reacts to current error)
    ki = 0.01 # Integral gain (Accumulates past error, reduces steady-state error)
    kd = 0.05 # Derivative gain (Reacts to rate of change of error, dampens oscillations)
    pid_controller = PID(kp, ki, kd)

    # --- Loop Variables ---
    frame_counter = 0
    loop_start_time = time.time()
    overall_fps = 0.0
    current_speed_limit = DEFAULT_MAX_SPEED
    speed_motor_requested = 0
    last_speed_sent = 0

    print("\nStarting main loop...")
    print(f"Default Speed Limit: {current_speed_limit}. Use W/A/S/X for manual/auto control.")
    print(f"NCNN Inference using: {'Vulkan GPU' if net.opt.use_vulkan_compute else 'CPU Threads'}")
    print("Press 'q' in the display window to exit.")

    try:
        while True:
            frame_start_time = time.time()

            # 1. Capture frame and perform basic image processing
            picam()
            if frame is None:
                print("Warning: Failed to capture frame.")
                time.sleep(0.1) # Wait briefly before retrying
                continue

            # 2. Perform lane detection processing (warp, ROI, Hough lines)
            getPoints(frame.shape[1], frame.shape[0]) # Define perspective points based on frame size
            warp_result = warpImg() # Warps canny_edge, finds lines in roi_haha

            # 3. Calculate steering angle based on lane detection
            SteeringAngle() # Calculates cte_f based on roi_haha

            # 4. Perform traffic sign detection on the original frame
            # Ensure detect_signs function uses the global 'net'
            detected_signs = detect_signs_and_get_results(frame)

            # 5. Handle keyboard input for manual control and exiting
            # waitKey(1) is crucial for OpenCV windows to update and capture keys
            key = cv2.waitKey(1) & 0xFF
            handle_key_input(key) # Updates global speed_motor_requested
            if key == ord('q'):
                print("'q' pressed, exiting loop.")
                break

            # 6. Process detected signs to update speed limit and check for stop conditions
            stop_condition_met = False
            limit_sign_seen_this_frame = False
            new_limit_value = -1 # Default invalid value

            if detected_signs:
                # print(" Signs Detected:") # Optional: Less verbose console
                highest_conf_limit = -1
                highest_conf = 0.0
                for sign in detected_signs:
                    # print(f"  - {sign['name']}: {sign['confidence']:.2f}") # Optional verbose output
                    # Check for stop sign
                    if sign['name'] == "Stop Sign":
                        stop_condition_met = True
                        # print("  ** STOP Condition Met **")

                    # Check for speed limit signs
                    parsed_limit = parse_speed_limit(sign['name'])
                    if parsed_limit is not None:
                        limit_sign_seen_this_frame = True
                        # Keep track of the limit from the most confident detection this frame
                        if sign['confidence'] > highest_conf:
                            highest_conf = sign['confidence']
                            highest_conf_limit = parsed_limit

                # Use the limit from the most confident sign seen this frame
                if limit_sign_seen_this_frame:
                    new_limit_value = highest_conf_limit

            # Update the active speed limit rule
            if limit_sign_seen_this_frame:
                if new_limit_value != current_speed_limit and new_limit_value != -1:
                    print(f"** Speed Limit Updated: {new_limit_value} (from sign) **")
                    current_speed_limit = new_limit_value
            else:
                # If no speed limit sign was seen, revert to default
                if current_speed_limit != DEFAULT_MAX_SPEED:
                    print(f"** No limit sign detected, resetting to default: {DEFAULT_MAX_SPEED} **")
                    current_speed_limit = DEFAULT_MAX_SPEED

            # 7. Determine the final speed command based on requests, limits, and stop conditions
            final_speed_command = speed_motor_requested # Start with user/auto request
            final_speed_command = min(final_speed_command, current_speed_limit) # Enforce speed limit
            if stop_condition_met:
                final_speed_command = 0 # Override speed if stop sign detected
            # Ensure speed is not negative
            final_speed_command = max(0, final_speed_command)

            # Update the last sent speed (used for display/logging)
            last_speed_sent = final_speed_command

            # 8. Calculate the steering command using the PID controller
            # Input is the Cross Track Error (cte_f), output is the servo angle
            steering = pid_controller.update(cte_f)

            # 9. Send Steering and Speed data to ESP32 via Serial (Binary Format)
            try:
                # Ensure values are integers within the expected range for ESP32
                steer_cmd = int(round(steering))
                speed_cmd = int(round(last_speed_sent))

                # Pack as two signed 16-bit integers (short), little-endian ('<')
                # Frame with '<' and '>'
                data_to_send = b'<' + struct.pack('<hh', steer_cmd, speed_cmd) + b'>'

                # --- Debug Print before sending ---
                # print(f"Sending Binary Data: {data_to_send} (Steer:{steer_cmd}, Speed:{speed_cmd})")
                # --- End Debug Print ---

                bytes_written = ser.write(data_to_send)
                ser.flush() # Try to ensure data is sent physically

                # Optional: Check if expected number of bytes were written
                # if bytes_written != len(data_to_send):
                #     print(f"Warning: Serial write error. Expected {len(data_to_send)} bytes, wrote {bytes_written}")

            except serial.SerialException as se:
                print(f"Serial Write Error: {se}. Attempting to continue...")
                # Consider adding logic to try reconnecting serial if errors persist
            except Exception as e:
                print(f"Error during serial write: {e}")


            # 10. Calculate and Print FPS / Status to Console
            frame_counter += 1
            current_loop_time = time.time()
            elapsed_time_total = current_loop_time - loop_start_time
            frame_time = current_loop_time - frame_start_time
            instant_fps = 1.0 / frame_time if frame_time > 0 else 0

            if elapsed_time_total > 1.0: # Update overall FPS roughly every second
                 overall_fps = frame_counter / elapsed_time_total

            # Print status less frequently or conditionally to reduce console spam
            if frame_counter % 10 == 0: # Print every 10 frames
                 print(f" Status: Limit={current_speed_limit} | Req={speed_motor_requested} | Sent={last_speed_sent} | Steer={steering} | CTE={cte_f:.2f}")
                 print(f" FPS: {instant_fps:.1f} (Avg: {overall_fps:.1f})")
                 print("-" * 20)


            # 11. Display Processed Image (Lane Detection ROI)
            # This window is needed for cv2.waitKey() to function
            if warp_result is not None and roi_haha is not None:
                # Convert grayscale ROI back to BGR for displaying text
                display_roi = cv2.cvtColor(roi_haha, cv2.COLOR_GRAY2BGR)

                # Draw status text on the display image
                cv2.putText(display_roi, f"FPS: {instant_fps:.1f}", (display_roi.shape[1] - 120, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
                cv2.putText(display_roi, f"CTE: {cte_f:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(display_roi, f"Steer: {steering}", (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(display_roi, f"Speed Sent: {last_speed_sent}", (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
                cv2.putText(display_roi, f"Speed Limit: {current_speed_limit}", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

                # Indicate if stop condition is met
                if stop_condition_met:
                    cv2.putText(display_roi, "STOP", (display_roi.shape[1] // 2 - 40 , display_roi.shape[0] // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)

                # Show the image
                cv2.imshow("Lane Detection ROI", display_roi)
            else:
                 # If warping failed, show the original frame maybe?
                 display_frame = frame.copy()
                 cv2.putText(display_frame, "Lane Warp Failed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                 cv2.imshow("Lane Detection ROI", display_frame) # Show something so waitKey works

    except KeyboardInterrupt:
        print("\nCtrl+C detected. Exiting gracefully...")
    except Exception as e:
        print(f"\nAn unexpected error occurred in the main loop: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
    finally:
        # --- Cleanup ---
        print("\nCleaning up resources...")

        # Send stop command to ESP32
        if ser and ser.is_open:
            try:
                print("Sending STOP command (Steer=90, Speed=0) to ESP32...")
                stop_data = b'<' + struct.pack('<hh', 90, 0) + b'>'
                ser.write(stop_data)
                ser.flush()
                time.sleep(0.1) # Give ESP32 time to process
                ser.close()
                print("Serial port closed.")
            except Exception as e:
                print(f"Error sending stop command or closing serial: {e}")

        # Stop camera
        if 'piCam' in globals() and piCam.started:
            try:
                piCam.stop()
                print("Picamera2 stopped.")
            except Exception as e:
                print(f"Error stopping Picamera2: {e}")

        # Close OpenCV windows
        cv2.destroyAllWindows()
        print("OpenCV windows closed.")
        print("Cleanup complete. Exiting.")

if __name__ == "__main__":
    main()