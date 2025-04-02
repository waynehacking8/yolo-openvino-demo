# 使用包含 CUDA 12.4 和 cuDNN 的 NVIDIA 官方映像檔 (基於 Ubuntu 22.04)
# -devel 版本包含編譯工具，有助於安裝某些 Python 套件
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04

# 設定環境變數，避免 apt-get 安裝過程中出現互動式提示
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC

# 設定容器內的工作目錄
WORKDIR /app

# 安裝基礎系統依賴項
# 更新 apt 套件列表並安裝必要的套件
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    wget \
    python3 \
    python3-pip \
    python3-dev \
    # OpenCV headless 所需的依賴項
    libgl1-mesa-glx \
    libglib2.0-0 \
    # 清理 apt 快取以減少映像檔大小
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 升級 pip
RUN python3 -m pip install --no-cache-dir --upgrade pip

# 複製 requirements.txt 文件
# 先複製此文件可以利用 Docker 的快取機制，如果 requirements.txt 沒有變更，就不會重新執行 pip install
COPY requirements.txt .

# 安裝 Python 依賴項
# 確保 requirements.txt 包含所有必要的套件及其兼容版本
# (例如: torch+cu124, openvino>=2024.6, nvidia-tensorrt, ultralytics, opencv-python-headless, nncf 等)
RUN echo "Installing Python dependencies from requirements.txt..." && \
    pip install --no-cache-dir -r requirements.txt && \
    echo "Dependency installation complete."

# 複製專案中的其餘程式碼到工作目錄
# .dockerignore 文件會排除不需要複製的文件
COPY . .

# (可選) 如果應用程式需要監聽端口，可以在這裡聲明
# EXPOSE 8080

# 設定容器啟動時的預設命令
# 這裡設為執行主腳本並顯示幫助訊息
CMD ["python3", "src/simple_demo.py", "--help"]