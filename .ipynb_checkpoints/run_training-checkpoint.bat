@echo off

echo Running train_prostate.py...
python train_prostate.py

echo Running train_fundus.py...
python train_fundus.py

echo Running train_prostate.py again...
python train_prostate.py

echo All training scripts have finished executing.
pause
