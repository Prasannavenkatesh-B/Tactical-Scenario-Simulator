@echo off
title DRDO Tactical Scenario Simulator (TSS) - 2D Interactive Simulator
cd /d "C:\Users\bpras\Desktop\TSS"
echo ===================================================================
echo     DRDO TACTICAL SCENARIO SIMULATOR (TSS) - 2D OPERATIONAL UI
echo ===================================================================
echo.
echo Launching Interactive 2D Simulator (Level 5 Combined Arms)...
echo Controls:
echo   - [SPACEBAR]   : Play / Pause Simulation
echo   - [RIGHT ARROW]: Step Forward 1 frame
echo   - [LEFT ARROW] : Reset Simulation
echo   - [W]          : Toggle Weapon Engagement Zones (WEZ)
echo   - [S]          : Toggle Radar Detection Cones
echo   - [T]          : Toggle Flight Trajectory Trails
echo   - [1-9] / Tab  : Manual Steering Override
echo.
python scripts\run_ui.py --level 5
if %ERRORLEVEL% NEQ 0 (
    echo.
    echo [!] Simulation closed or exited with error code %ERRORLEVEL%.
    pause
)
