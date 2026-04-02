import os
import sys
import argparse
import glob
import time

import cv2
import numpy as np
from ultralytics import YOLO

# Define and parse user input arguments

parser = argparse.ArgumentParser()
parser.add_argument('--model', help='Path to YOLO model file (example: "runs/detect/train/weights/best.pt")',
                    required=True)
parser.add_argument('--source', help='Image source, can be image file ("test.jpg"), \
                    image folder ("test_dir"), video file ("testvid.mp4"), or index of USB camera ("usb0")', 
                    required=True)
parser.add_argument('--thresh', help='Minimum confidence threshold for displaying detected objects (example: "0.4")',
                    default=0.5)
parser.add_argument('--resolution', help='Resolution in WxH to display inference results at (example: "640x480"), \
                    otherwise, match source resolution',
                    default=None)
parser.add_argument('--record', help='Record results from video or webcam and save it as "demo1.avi". Must specify --resolution argument to record.',
                    action='store_true')

args = parser.parse_args()


# Parse user inputs
model_path = args.model
img_source = args.source
min_thresh = args.thresh
user_res = args.resolution
record = args.record

# Check if model file exists and is valid
if (not os.path.exists(model_path)):
    print('ERROR: Model path is invalid or model was not found. Make sure the model filename was entered correctly.')
    sys.exit(0)

# Load the model into memory and get labemap
model = YOLO(model_path, task='detect')
labels = model.names

# Parse input to determine if image source is a file, folder, video, or USB camera
img_ext_list = ['.jpg','.JPG','.jpeg','.JPEG','.png','.PNG','.bmp','.BMP']
vid_ext_list = ['.avi','.mov','.mp4','.mkv','.wmv']

if os.path.isdir(img_source):
    source_type = 'folder'
elif os.path.isfile(img_source):
    _, ext = os.path.splitext(img_source)
    if ext in img_ext_list:
        source_type = 'image'
    elif ext in vid_ext_list:
        source_type = 'video'
    else:
        print(f'File extension {ext} is not supported.')
        sys.exit(0)
elif 'usb' in img_source:
    source_type = 'usb'
    usb_idx = int(img_source[3:])
elif 'picamera' in img_source:
    source_type = 'picamera'
    picam_idx = int(img_source[8:])
else:
    print(f'Input {img_source} is invalid. Please try again.')
    sys.exit(0)

# Parse user-specified display resolution
resize = False
if user_res:
    resize = True
    resW, resH = int(user_res.split('x')[0]), int(user_res.split('x')[1])

# Check if recording is valid and set up recording
if record:
    if source_type not in ['video','usb']:
        print('Recording only works for video and camera sources. Please try again.')
        sys.exit(0)
    if not user_res:
        print('Please specify resolution to record video at.')
        sys.exit(0)
    
    # Set up recording
    record_name = 'demo1.avi'
    record_fps = 30
    recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW,resH))

# Load or initialize image source
if source_type == 'image':
    imgs_list = [img_source]
elif source_type == 'folder':
    imgs_list = []
    filelist = glob.glob(img_source + '/*')
    for file in filelist:
        _, file_ext = os.path.splitext(file)
        if file_ext in img_ext_list:
            imgs_list.append(file)
elif source_type == 'video' or source_type == 'usb':

    if source_type == 'video': cap_arg = img_source
    elif source_type == 'usb': cap_arg = usb_idx
    cap = cv2.VideoCapture(cap_arg)

    # Set camera or video resolution if specified by user
    if user_res:
        ret = cap.set(3, resW)
        ret = cap.set(4, resH)

elif source_type == 'picamera':
    from picamera2 import Picamera2
    cap = Picamera2()
    cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
    cap.start()

# Set bounding box colors (using the Tableu 10 color scheme)
bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106), 
              (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

# Initialize control and status variables
avg_frame_rate = 0
frame_rate_buffer = []
fps_avg_len = 200
img_count = 0

# Rock Paper Scissors stuff
# --- RPS GAME SETUP ---
import random # Add this at the very top of your script

# Define what beats what. Make sure these match your YOLO labels exactly!
# If your labels are '0', '1', '2', change the keys below to '0', '1', etc.
rules = {
    'Rock': 'Scissors',
    'Paper': 'Rock',
    'Scissors': 'Paper'
}

cpu_images = {
    'Rock': 'CPU choice img/Rock.jpg',
    'Paper': 'CPU choice img/Papper.avif',
    'Scissors': 'CPU choice img/Scissors.webp'
}

game_state = "WAITING"  # Options: WAITING, COUNTING, RESULTS
game_timer = 0
cpu_choice = None
user_choice = None
winner_text = ""
# ----------------------

# Begin inference loop
while True:

    t_start = time.perf_counter()

    # Load frame from image source
    if source_type == 'image' or source_type == 'folder': # If source is image or image folder, load the image using its filename
        if img_count >= len(imgs_list):
            print('All images have been processed. Exiting program.')
            sys.exit(0)
        img_filename = imgs_list[img_count]
        frame = cv2.imread(img_filename)
        img_count = img_count + 1
    
    elif source_type == 'video': # If source is a video, load next frame from video file
        ret, frame = cap.read()
        if not ret:
            print('Reached end of the video file. Exiting program.')
            break
    
    elif source_type == 'usb': # If source is a USB camera, grab frame from camera
        ret, frame = cap.read()
        if (frame is None) or (not ret):
            print('Unable to read frames from the camera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    elif source_type == 'picamera': # If source is a Picamera, grab frames using picamera interface
        frame_bgra = cap.capture_array()
        frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
        if (frame is None):
            print('Unable to read frames from the Picamera. This indicates the camera is disconnected or not working. Exiting program.')
            break

    # Resize frame to desired display resolution
    if resize == True:
        frame = cv2.resize(frame,(resW,resH))

    # Run inference on frame
    results = model(frame, verbose=False)

    # Extract results
    detections = results[0].boxes

    # Initialize variable for basic object counting example
    object_count = 0

  # IMPORTANT: Define resW/resH if they don't exist (safety)
    if not resize:
        resH, resW, _ = frame.shape

    # Go through each detection and get bbox coords, confidence, and class
    # --- DETECTION LOOP ---
    for i in range(len(detections)):
        classidx = int(detections[i].cls.item())
        classname = labels[classidx]
        conf = detections[i].conf.item()

        if conf > min_thresh: # Use your threshold variable here
            xyxy = detections[i].xyxy.cpu().numpy().squeeze().astype(int)
            xmin, ymin, xmax, ymax = xyxy
            
            # Draw visual feedback
            color = bbox_colors[classidx % 10]
            cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), color, 2)
            
            # Capture the player's move
            current_user_move = classname
            object_count += 1

    # --- GAME LOGIC ---
    current_time = time.perf_counter()

    if game_state == "COUNTING":
        countdown = int(game_timer - current_time)
        cv2.putText(frame, f"READY... {countdown}", (resW//3, resH//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, (0, 255, 255), 10)
        
        if countdown <= 0:
            game_state = "RESULTS"
            cpu_choice = random.choice(list(rules.keys()))
            
            # Use 'current_user_move' captured from the loop above
            user_choice = current_user_move if 'current_user_move' in locals() else "None"
            
            if user_choice == cpu_choice:
                winner_text = "TIE :/"
            elif rules.get(user_choice) == cpu_choice:
                winner_text = "YOU WIN :D"
            else:
                winner_text = "YOU LOSE ;("

    elif game_state == "RESULTS":

        # Get the results text
        results_text = f"YOU: {user_choice.upper()} | CPU: {cpu_choice.upper()}"
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(results_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
        
        # Define padding around the text
        padding = 10
        
        # Calculate box coordinates
        box_x1 = 10 - padding
        box_y1 = 100 - text_height - padding
        box_x2 = 10 + text_width + padding
        box_y2 = 100 + baseline + padding
        
        # Draw a semi-transparent overlay box sized to the text
        overlay = frame.copy()
        cv2.rectangle(overlay, (box_x1, box_y1), (box_x2, box_y2), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
        
        cv2.putText(frame, results_text, 
                    (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

      # Color the winner text: Green for Win, Red for Loss
        if winner_text == "YOU WIN :D": result_color = (0,255,0)
        elif winner_text == "TIE :/": result_color = (0, 255, 255)
        else : result_color = (0,0,255)

        cv2.putText(frame, winner_text, (resW//3, 200), 
                    cv2.FONT_HERSHEY_SIMPLEX, 3, result_color , 6)
        cv2.putText(frame, "Press 'SPACE' to Play Again", (10, resH - 20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

        # Display CPU choice image below results
        if cpu_choice in cpu_images:
            img_path = cpu_images[cpu_choice]
            cpu_img = cv2.imread(img_path)
            if cpu_img is not None:
                # Resize to 300x300
                cpu_img = cv2.resize(cpu_img, (300, 300))
                # Position below winner_text, to the left
                img_h, img_w = cpu_img.shape[:2]
                start_x = 30  # Left aligned
                start_y = 150  # Adjust as needed
                # Ensure it fits in frame
                if start_y + img_h <= resH and start_x >= 0 and start_x + img_w <= resW:
                    frame[start_y:start_y+img_h, start_x:start_x+img_w] = cpu_img

    elif game_state == "WAITING":
        cv2.putText(frame, "PRESS 'SPACE' TO PLAY", (resW//4, resH//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 8) # Shadow
        cv2.putText(frame, "PRESS 'SPACE' TO PLAY", (resW//4, resH//2), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
#----------------------------------------------------------------------------------------

    cv2.imshow('YOLO detection results',frame) # Display image
    if record: recorder.write(frame)

    # If inferencing on individual images, wait for user keypress before moving to next image. Otherwise, wait 5ms before moving to next frame.
    if source_type == 'image' or source_type == 'folder':
        key = cv2.waitKey()
    elif source_type == 'video' or source_type == 'usb' or source_type == 'picamera':
        key = cv2.waitKey(5)
    
    if key == ord('q') or key == ord('Q'): # Press 'q' to quit
        break
    elif key == ord('s') or key == ord('S'): # Press 's' to pause inference
        cv2.waitKey()
    elif key == ord('p') or key == ord('P'): # Press 'p' to save a picture of results on this frame
        cv2.imwrite('capture.png',frame)

    # space to start game countdown
    if key == ord(' '): # Spacebar
        game_state = "COUNTING"
        game_timer = time.perf_counter() + 4 # 3 second countdown + 1s buffer
        current_user_move = "None" # Reset move
    #-------------------------------------------------------------------

    # Calculate FPS for this frame
    t_stop = time.perf_counter()
    frame_rate_calc = float(1/(t_stop - t_start))

    # Append FPS result to frame_rate_buffer (for finding average FPS over multiple frames)
    if len(frame_rate_buffer) >= fps_avg_len:
        temp = frame_rate_buffer.pop(0)
        frame_rate_buffer.append(frame_rate_calc)
    else:
        frame_rate_buffer.append(frame_rate_calc)

    # Calculate average FPS for past frames
    avg_frame_rate = np.mean(frame_rate_buffer)


# Clean up
print(f'Average pipeline FPS: {avg_frame_rate:.2f}')
if source_type == 'video' or source_type == 'usb':
    cap.release()
elif source_type == 'picamera':
    cap.stop()
if record: recorder.release()
cv2.destroyAllWindows()
