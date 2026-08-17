python -m venv tensor_flow_env
source tensor_flow_env/bin/activate
python -m pip install --upgrade pip
python -c "import tensorflow as tf; print('TensorFlow version:', tf.__version__)"
# 数据处理
pip install numpy pandas matplotlib seaborn

# 科学计算
pip install scipy scikit-learn

# 图像处理
pip install pillow opencv-python

# 进度条和实用工具
pip install tqdm

# 创建完整的要求文件
cat > requirements.txt << EOF
tensorflow>=2.12.0
numpy>=1.21.0
pandas>=1.3.0
matplotlib>=3.5.0
seaborn>=0.11.0
scikit-learn>=1.0.0
jupyter>=1.0.0
tqdm>=4.60.0
pillow>=8.0.0
EOF

# 批量安装
pip install -r requirements.txt
