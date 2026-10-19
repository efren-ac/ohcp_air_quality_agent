@echo off
chcp 65001 >nul
cd /d "%~dp0"
if not exist ".env" copy ".env.example" ".env" >nul
echo ================================================================
echo OHCP 空气质量时序分析 Agent
echo ================================================================
echo 请确认已在 Anaconda Prompt 中激活 DB 环境。
python check_environment.py
if errorlevel 1 (
  echo.
  echo 环境检查存在问题，请先查看上方提示。
  pause
  exit /b 1
)
python main.py
pause
