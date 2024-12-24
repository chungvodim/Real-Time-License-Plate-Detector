import cv2
from inference import LicensePlate

license = LicensePlate()

cap = cv2.VideoCapture(r"sample_vids\sample_1.mp4")

# Check if the video stream is opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream.")
    exit()

# Get the width and height of frames
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the codec and create VideoWriter object for MP4
out = cv2.VideoWriter('output_video.mp4', cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))

while True:
    ret, frame = cap.read()

    if not ret:
        break
    
    license_frame = license.detect_license(frame)
    # Write the frame into the file 'output_video.mp4'
    out.write(license_frame)

# Release everything when the job is finished
cap.release()
out.release()
cv2.destroyAllWindows()
