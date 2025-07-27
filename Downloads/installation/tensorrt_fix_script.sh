#!/bin/bash

echo "=== FINDING AND INSTALLING CORRECT TENSORRT PACKAGES ==="
echo

# First, let's see what TensorRT packages are available
echo "1. SEARCHING FOR AVAILABLE TENSORRT PACKAGES:"
echo "---------------------------------------------"
apt-cache search tensorrt | grep -E "(python|libnvinfer)" | head -10
echo

echo "2. SEARCHING FOR LIBNVINFER PACKAGES:"
echo "------------------------------------"  
apt-cache search libnvinfer | head -10
echo

echo "3. CHECKING WHAT'S ALREADY INSTALLED:"
echo "------------------------------------"
dpkg -l | grep -E "(tensorrt|libnvinfer)" | head -10
echo

# Try to find the correct package names
echo "4. FINDING EXACT PACKAGE NAMES:"
echo "-------------------------------"
echo "Available python3-libnvinfer packages:"
apt list --installed 2>/dev/null | grep libnvinfer || echo "None installed"
apt-cache search python3-libnvinfer
echo

echo "Available tensorrt packages:"
apt-cache search python.*tensorrt
echo

# Let's check what version of TensorRT we have
echo "5. CHECKING TENSORRT VERSION FROM LIBRARIES:"
echo "-------------------------------------------"
if [ -f /usr/lib/aarch64-linux-gnu/libnvinfer.so.8 ]; then
    echo "TensorRT 8.x libraries found"
    ls -la /usr/lib/aarch64-linux-gnu/libnvinfer* | head -5
else
    echo "Checking for other TensorRT versions..."
    ls -la /usr/lib/aarch64-linux-gnu/libnvinfer* 2>/dev/null | head -5 || echo "No TensorRT libraries found"
fi
echo

# Check if there are any python bindings in system packages
echo "6. CHECKING SYSTEM PYTHON PACKAGES:"
echo "----------------------------------"
find /usr/lib/python3*/dist-packages -name "*tensorrt*" 2>/dev/null || echo "No TensorRT in system packages"
find /usr/lib/python3*/dist-packages -name "*nvinfer*" 2>/dev/null || echo "No nvinfer in system packages"
echo

echo "=== INSTALLATION ATTEMPTS ==="
echo

# Try different package combinations based on what we found
echo "Attempting to install TensorRT Python bindings..."

# Method 1: Try standard names for TensorRT 8.x
echo "Method 1: Installing standard TensorRT 8.x packages..."
sudo apt update
sudo apt install -y python3-libnvinfer python3-libnvinfer-dev libnvinfer8 libnvinfer-dev 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ Standard packages installed successfully"
else
    echo "❌ Standard packages failed"
fi

# Method 2: Try TensorRT 8.5 specific packages (matching your libraries)
echo "Method 2: Trying TensorRT 8.5 specific packages..."
sudo apt install -y libnvinfer8 libnvinfer-dev libnvinfer-plugin8 libnvinfer-plugin-dev 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ TensorRT 8.5 libraries installed"
else
    echo "❌ TensorRT 8.5 libraries failed"
fi

# Method 3: Install Python bindings manually if packages don't exist
echo "Method 3: Checking if Python bindings need manual setup..."

# Create symbolic links if needed
if [ -d /usr/lib/python3.8/dist-packages ] && [ ! -d /usr/lib/python3.8/dist-packages/tensorrt ]; then
    echo "Setting up Python 3.8 TensorRT links..."
    sudo mkdir -p /usr/lib/python3.8/dist-packages/tensorrt 2>/dev/null
fi

if [ -d /usr/lib/python3.10/dist-packages ] && [ ! -d /usr/lib/python3.10/dist-packages/tensorrt ]; then
    echo "Setting up Python 3.10 TensorRT links..."
    sudo mkdir -p /usr/lib/python3.10/dist-packages/tensorrt 2>/dev/null
fi

# Method 4: Try installing from nvidia-jetpack meta package
echo "Method 4: Installing JetPack components..."
sudo apt install -y nvidia-jetpack 2>/dev/null
if [ $? -eq 0 ]; then
    echo "✅ JetPack meta package installed"
else
    echo "❌ JetPack meta package not available"
fi

echo
echo "=== TESTING INSTALLATION ==="
echo

# Test Python import
echo "Testing TensorRT Python import..."
python3 -c "
import sys
# Add common system paths
sys.path.extend([
    '/usr/lib/python3/dist-packages',
    '/usr/lib/python3.8/dist-packages', 
    '/usr/lib/python3.10/dist-packages',
    '/usr/local/lib/python3.8/dist-packages',
    '/usr/local/lib/python3.10/dist-packages'
])

try:
    import tensorrt as trt
    print(f'✅ SUCCESS: TensorRT version {trt.__version__}')
    print(f'✅ TensorRT Builder available: {trt.Builder is not None}')
    print(f'✅ TensorRT path: {trt.__file__}')
except ImportError as e:
    print(f'❌ FAILED: {e}')
    print('Available paths:')
    for path in sys.path:
        print(f'  {path}')
"

echo
echo "=== ALTERNATIVE SOLUTIONS ==="
echo

if ! python3 -c "import tensorrt" 2>/dev/null; then
    echo "If TensorRT Python still doesn't work, try these alternatives:"
    echo
    echo "1. Manual wheel installation:"
    echo "   Find TensorRT wheel in JetPack and install manually"
    echo
    echo "2. Use ONNX instead of TensorRT:"
    echo "   model.export(format='onnx', device=0)"
    echo "   # Then use ONNX runtime for inference"
    echo
    echo "3. Run without optimization:"
    echo "   # Your code will work fine with regular PyTorch models"
    echo "   # TensorRT is just for performance optimization"
    echo
    echo "4. Check JetPack SDK installation:"
    echo "   sudo apt install nvidia-jetpack-dev"
fi