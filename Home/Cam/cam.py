import cv2

def gstreamer_pipeline(
    capture_width=1280,
    capture_height=720,
    framerate=30,
    flip_method=0
):
    # Fixed pipeline - removed NVMM memory and BGRx format issues
    return (
        f"v4l2src device=/dev/video0 ! "
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"jpegdec ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, format=BGR ! "  # Changed from BGRx to BGR, removed NVMM
        f"videoconvert ! "
        f"appsink drop=1"  # Added drop=1 to prevent buffer overflow
    )

# Alternative software-only pipeline (if hardware acceleration fails)
def gstreamer_pipeline_software(
    capture_width=1280,
    capture_height=720,
    framerate=30
):
    return (
        f"v4l2src device=/dev/video0 ! "
        f"image/jpeg, width={capture_width}, height={capture_height}, framerate={framerate}/1 ! "
        f"jpegdec ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink drop=1"
    )

# Initialize camera with hardware acceleration
print("Trying hardware-accelerated pipeline...")
cap = cv2.VideoCapture(gstreamer_pipeline(), cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Hardware acceleration failed. Trying software pipeline...")
    cap = cv2.VideoCapture(gstreamer_pipeline_software(), cv2.CAP_GSTREAMER)
    
    if not cap.isOpened():
        print("GStreamer failed. Trying basic OpenCV...")
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ All methods failed!")
            exit(1)

print("✅ Camera initialized successfully. Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("⚠️ Failed to grab frame")
        continue
    
    # Flip frame horizontally
    frame = cv2.flip(frame, 1)
    
    cv2.imshow("DJI Cam - Fixed", frame)
    
    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()