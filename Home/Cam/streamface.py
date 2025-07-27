import cv2

# Load the face detection classifier (lightweight Haar cascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# OpenCV capture from USB webcam
cap = cv2.VideoCapture(0)

# Check if camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video capture")
    exit()

# Set resolution and framerate (optional)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
cap.set(cv2.CAP_PROP_FPS, 30)

# GStreamer pipeline to send video via UDP to host PC
host_ip = "192.168.1.100"  # replace with your host PC's IP
gst_str = (
    f'appsrc ! videoconvert ! x264enc tune=zerolatency bitrate=1000 speed-preset=superfast '
    f'! rtph264pay config-interval=1 pt=96 ! udpsink host={host_ip} port=5000'
)

# OpenCV VideoWriter for sending via GStreamer
out = cv2.VideoWriter(gst_str, cv2.CAP_GSTREAMER, 0, 30, (1280, 720))

if not out.isOpened():
    print("Error: Could not open video writer")
    exit()

print("Streaming with face detection... Press Ctrl+C to stop")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame")
            break
        
        # Convert to grayscale for face detection (faster processing)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces (optimized parameters for speed)
        faces = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Draw rectangles around detected faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Optional: Add a label
            cv2.putText(frame, 'Face', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Send the frame with face detection overlay to the stream
        out.write(frame)

except KeyboardInterrupt:
    print("Streaming stopped")

finally:
    cap.release()
    out.release()