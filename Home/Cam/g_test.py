#!/usr/bin/env python3
import cv2
import sys

print("=== Simple Camera Test ===")

# Test 1: Basic camera access
print("1. Testing basic camera access...")
cap = cv2.VideoCapture(0)
if cap.isOpened():
    print("✓ Basic camera access works")
    ret, frame = cap.read()
    if ret:
        height, width, channels = frame.shape
        print(f"✓ Frame captured: {width}x{height}, {channels} channels")
    else:
        print("✗ Could not read frame")
    cap.release()
else:
    print("✗ Basic camera access failed")
    sys.exit(1)

# Test 2: GStreamer availability
print("\n2. Testing GStreamer availability...")
try:
    # Try with explicit backend specification
    cap_gst = cv2.VideoCapture()
    if cap_gst.open("videotestsrc ! videoconvert ! appsink", cv2.CAP_GSTREAMER):
        print("✓ GStreamer backend works")
        cap_gst.release()
    else:
        print("✗ GStreamer backend failed")
        # Try alternative test
        print("   Trying alternative GStreamer test...")
        cap_gst2 = cv2.VideoCapture("videotestsrc num-buffers=1 ! videoconvert ! appsink drop=1", cv2.CAP_GSTREAMER)
        if cap_gst2.isOpened():
            print("✓ Alternative GStreamer test works")
            cap_gst2.release()
        else:
            print("✗ Alternative GStreamer test failed")
except Exception as e:
    print(f"✗ GStreamer error: {e}")

# Test 3: Simple GStreamer camera
print("\n3. Testing simple GStreamer camera...")
pipelines_to_try = [
    "v4l2src device=/dev/video0 ! videoconvert ! appsink drop=1",
    "v4l2src device=/dev/video0 ! video/x-raw,format=YUY2,width=640,height=480 ! videoconvert ! appsink",
    "v4l2src device=/dev/video0 ! jpegdec ! videoconvert ! appsink"
]

for i, pipeline in enumerate(pipelines_to_try):
    print(f"\n3.{i+1}. Testing pipeline: {pipeline}")
    try:
        cap_simple = cv2.VideoCapture()
        if cap_simple.open(pipeline, cv2.CAP_GSTREAMER):
            print("✓ Pipeline opened successfully")
            ret, frame = cap_simple.read()
            if ret:
                print(f"✓ GStreamer frame captured: {frame.shape}")
                cap_simple.release()
                break
            else:
                print("✗ Could not read frame")
            cap_simple.release()
        else:
            print("✗ Pipeline failed to open")
    except Exception as e:
        print(f"✗ Pipeline error: {e}")
        
# Test 4: Check GStreamer plugins
print("\n4. Testing specific GStreamer elements...")
import subprocess
elements_to_check = ['v4l2src', 'videoconvert', 'appsink']
for element in elements_to_check:
    try:
        result = subprocess.run(['gst-inspect-1.0', element], 
                              capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print(f"✓ {element} plugin available")
        else:
            print(f"✗ {element} plugin missing")
    except Exception as e:
        print(f"✗ Error checking {element}: {e}")

print("\nTest complete!")