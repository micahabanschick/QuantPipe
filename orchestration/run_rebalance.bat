@echo off
cd /d "C:\Users\micha\Documents\Programming\QuantPipe"
".venv\Scripts\python.exe" orchestration\rebalance.py --broker paper >> logs\rebalance.log 2>&1
