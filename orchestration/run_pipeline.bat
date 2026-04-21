@echo off
cd /d "C:\Users\micha\Documents\Programming\QuantPipe"
".venv\Scripts\python.exe" orchestration\run_pipeline.py >> logs\pipeline.log 2>&1
