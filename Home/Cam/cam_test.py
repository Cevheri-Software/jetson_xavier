import cv2
import time

# GStreamer pipeline for Jetson hardware-accelerated MJPEG decoding
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

# Initialize GStreamer capture
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Failed to open camera with GStreamer. Falling back to basic capture...")
    # Optional fallback (use only if GStreamer fails)
    cap = cv2.VideoCapture(0)

# Set manual exposure or other properties if needed:
# cap.set(cv2.CAP_PROP_EXPOSURE, -4)

print("✅ Camera initialized. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        print(" Failed to grab frame")
        break
    
    # Flip frame horizontally
    frame = cv2.flip(frame, 1) 
    
    cv2.imshow("DJI Cam - Low Latency", frame)

    # # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
