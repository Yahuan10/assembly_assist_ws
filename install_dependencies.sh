#!/bin/bash

echo "安装系统依赖（APT）..."
sudo apt update
sudo apt install -y python3-pip python3-tk libgl1

echo "创建 venv（可选）..."
python3 -m venv venv
source venv/bin/activate

echo "安装 Python requirements..."
pip install --upgrade pip
pip install -r requirements.txt

echo "所有依赖安装完成！"
