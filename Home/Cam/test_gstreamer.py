import cv2
import numpy as np

HOST_IP = '192.168.1.100'  # HOST PC IP
PORT = 5000

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0
):
    return (
        f"v4l2src device=/dev/video0 ! "
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"jpegdec ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw(memory:NVMM), width={capture_width}, height={capture_height}, format=BGRx ! "
        f"videoconvert ! appsink"
    )

# Open the camera capture with GStreamer
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera with GStreamer. Falling back to basic capture...")
    cap = cv2.VideoCapture(0)

# Setup GStreamer pipeline for streaming to host PC via UDP
gst_out_pipeline = (
    f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=1000 speed-preset=superfast ! '
    f'rtph264pay ! udpsink host={HOST_IP} port={PORT}'
)

# VideoWriter for streaming (make sure width,height match capture size)
out = cv2.VideoWriter(gst_out_pipeline, cv2.CAP_GSTREAMER, 0, 30, (1280, 720), True)

if not out.isOpened():
    print("Error: Could not open VideoWriter for streaming")
    cap.release()
    exit()

print("Streaming started. Press Ctrl+C to stop.")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Flip frame horizontally
        frame = cv2.flip(frame, 1)

        # Write frame to GStreamer pipeline (stream it)
        out.write(frame)

except KeyboardInterrupt:
    print("Streaming stopped by user.")

cap.release()
out.release()
cv2.destroyAllWindows()
