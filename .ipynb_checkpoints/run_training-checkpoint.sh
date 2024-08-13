#!/bin/bash

# 运行第一个 train_prostate.py
echo "Running train_prostate.py..."
python3 train_prostate.py

# 运行 train_fundus.py
echo "Running train_fundus.py..."
python3 train_fundus.py

# 再次运行 train_prostate.py
echo "Running train_prostate.py again..."
python3 train_prostate.py

echo "All training scripts have finished executing."
