@echo off
REM Run a short training run (synthetic data) to verify GPU + training loop
python src\train.py --config configs\small.json --batch 1 --use_amp --steps 10
pause
