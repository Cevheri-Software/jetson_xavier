#!/bin/bash

set -e
cd ~

echo "🔍 Mevcut OpenCV klasörleri kontrol ediliyor..."

if [[ ! -d "/home/cevheri/Cam"/opencv-python || ! -d "opencv_contrib" ]]; then
    echo "❌ 'opencv' veya 'opencv_contrib' klasörleri bulunamadı. Scripti iptal ediyorum."
    exit 1
fi

echo "✅ OpenCV kaynak klasörleri bulundu."

# Gerekli GStreamer paketlerini yükle
echo -e "\n📦 GStreamer ve bağımlılıkları yükleniyor...\n"
sudo apt update && sudo apt install -y \
  libgstreamer1.0-dev \
  libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-tools \
  gstreamer1.0-plugins-base \
  gstreamer1.0-plugins-good \
  gstreamer1.0-plugins-bad \
  gstreamer1.0-plugins-ugly \
  gstreamer1.0-libav \
  libcanberra-gtk3-module

# Derleme klasörüne geç
cd ~/opencv
mkdir -p build && cd build

# Eski CMake cache’i temizle
rm -rf CMakeCache.txt CMakeFiles/

# CMake yeniden yapılandırılıyor
echo -e "\n⚙️ CMake yapılandırması başlatılıyor...\n"

cmake -D CMAKE_BUILD_TYPE=RELEASE \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib/modules \
  -D WITH_CUDA=ON \
  -D WITH_GSTREAMER=ON \
  -D WITH_LIBV4L=ON \
  -D BUILD_opencv_python3=ON \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D WITH_QT=OFF \
  -D WITH_OPENGL=ON \
  ..

# Derleme işlemi
echo -e "\n🔨 make başlatılıyor...\n"
make -j$(nproc)

# Kurulum
echo -e "\n📥 OpenCV kuruluyor...\n"
sudo make install
sudo ldconfig

# Doğrulama
echo -e "\n✅ Kurulum tamamlandı. GStreamer desteği kontrol ediliyor...\n"
python3 -c "import cv2; print(cv2.getBuildInformation())" | grep -i gstreamer

