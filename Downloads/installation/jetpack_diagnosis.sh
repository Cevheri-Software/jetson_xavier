#!/bin/bash

echo "=== JETPACK AND TENSORRT DIAGNOSIS ==="
echo

# Check if this is a Jetson device
echo "1. CHECKING JETSON DEVICE INFO:"
echo "--------------------------------"
if [ -f /etc/nv_tegra_release ]; then
    echo "✓ Jetson device detected:"
    cat /etc/nv_tegra_release
    echo
else
    echo "✗ Not a Jetson device or tegra release file missing"
    echo
fi

# Check device model
if [ -f /proc/device-tree/model ]; then
    echo "Device Model: $(cat /proc/device-tree/model)"
    echo
fi

# Check L4T version
echo "2. CHECKING L4T VERSION:"
echo "------------------------"
if [ -f /etc/nv_tegra_release ]; then
    echo "L4T Version from tegra release:"
    cat /etc/nv_tegra_release | grep "R32\|R35\|R36"
fi

head -n 1 /etc/nv_tegra_release 2>/dev/null || echo "No tegra release found"
echo

# Check CUDA installation
echo "3. CHECKING CUDA INSTALLATION:"
echo "------------------------------"
if command -v nvcc &> /dev/null; then
    echo "✓ CUDA Compiler (nvcc) found:"
    nvcc --version | grep "release"
    echo
else
    echo "✗ nvcc not found in PATH"
    echo "Checking for CUDA in common locations..."
    if [ -d /usr/local/cuda ]; then
        echo "✓ CUDA found in /usr/local/cuda"
        /usr/local/cuda/bin/nvcc --version 2>/dev/null | grep "release" || echo "nvcc not executable"
    else
        echo "✗ CUDA not found in /usr/local/cuda"
    fi
    echo
fi

# Check cuDNN
echo "4. CHECKING cuDNN:"
echo "------------------"
if [ -f /usr/include/cudnn.h ]; then
    echo "✓ cuDNN header found:"
    cat /usr/include/cudnn.h | grep "#define CUDNN_MAJOR" -A 2 2>/dev/null
elif [ -f /usr/include/cudnn_version.h ]; then
    echo "✓ cuDNN version header found:"
    cat /usr/include/cudnn_version.h | grep "#define CUDNN_MAJOR" -A 2 2>/dev/null
else
    echo "✗ cuDNN headers not found"
fi
echo

# Check TensorRT installation
echo "5. CHECKING TENSORRT:"
echo "--------------------"

# Check TensorRT libraries
echo "TensorRT Libraries:"
if ls /usr/lib/aarch64-linux-gnu/libnvinfer* 2>/dev/null | head -5; then
    echo "✓ TensorRT libraries found"
else
    echo "✗ No TensorRT libraries found in /usr/lib/aarch64-linux-gnu/"
fi
echo

# Check TensorRT Python bindings
echo "TensorRT Python bindings:"
python3 -c "
try:
    import tensorrt as trt
    print(f'✓ TensorRT Python available: {trt.__version__}')
    print(f'  Builder version: {trt.Builder.get_plugin_registry().get_plugin_creator_list()}' if hasattr(trt.Builder, 'get_plugin_registry') else '  Basic TensorRT functions available')
except ImportError as e:
    print(f'✗ TensorRT Python not available: {e}')
    print('Checking system paths...')
    import sys
    sys.path.append('/usr/lib/python3/dist-packages')
    try:
        import tensorrt as trt
        print(f'✓ TensorRT found in system packages: {trt.__version__}')
    except ImportError:
        print('✗ TensorRT not found in system packages either')
"
echo

# Check installed packages
echo "6. CHECKING INSTALLED PACKAGES:"
echo "-------------------------------"
echo "NVIDIA/CUDA related packages:"
dpkg -l | grep -E "(cuda|nvidia|tensorrt|cudnn)" | head -10
echo

# Check if JetPack meta-package is installed
echo "JetPack meta-packages:"
dpkg -l | grep -i jetpack
echo

# Check NVIDIA container runtime
echo "7. CHECKING NVIDIA CONTAINER RUNTIME:"
echo "------------------------------------"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi found:"
    nvidia-smi | head -3
else
    echo "✗ nvidia-smi not found"
fi
echo

# Check environment variables
echo "8. CHECKING ENVIRONMENT VARIABLES:"
echo "----------------------------------"
echo "CUDA_HOME: ${CUDA_HOME:-'Not set'}"
echo "PATH (CUDA related): $(echo $PATH | tr ':' '\n' | grep cuda || echo 'No CUDA paths')"
echo "LD_LIBRARY_PATH (CUDA related): $(echo $LD_LIBRARY_PATH | tr ':' '\n' | grep cuda || echo 'No CUDA library paths')"
echo

# Check GPU status
echo "9. CHECKING GPU STATUS:"
echo "----------------------"
if [ -f /sys/class/devfreq/17000000.gv11b/cur_freq ]; then
    echo "✓ GPU detected (Jetson)"
    echo "GPU frequency: $(cat /sys/class/devfreq/17000000.gv11b/cur_freq) Hz"
elif [ -f /sys/kernel/debug/tegra_gpu/freq ]; then
    echo "✓ GPU detected (Tegra)"
    echo "GPU frequency: $(cat /sys/kernel/debug/tegra_gpu/freq)"
else
    echo "GPU status files not found (this might be normal)"
fi
echo

echo "=== RECOMMENDATIONS ==="
echo

# Provide recommendations based on findings
echo "To fix missing components:"
echo "1. If CUDA is missing: sudo apt install cuda-toolkit-*"
echo "2. If TensorRT is missing: sudo apt install python3-libnvinfer*"
echo "3. If cuDNN is missing: sudo apt install libcudnn8*"
echo "4. Add to ~/.bashrc if needed:"
echo "   export PATH=/usr/local/cuda/bin:\$PATH"
echo "   export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH"
echo "5. Restart jtop service: sudo systemctl restart jtop"
echo

echo "Run this to install missing TensorRT components:"
echo "sudo apt update && sudo apt install python3-libnvinfer python3-libnvinfer-dev libnvinfer8 libnvinfer-dev"