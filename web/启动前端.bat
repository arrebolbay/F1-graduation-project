@echo off
rem ============================================================
rem  F1 摩纳哥站进站策略优化系统 · 一键启动
rem  启动本地 Web 服务并自动打开浏览器(默认 http://127.0.0.1:8765/)
rem ============================================================
cd /d %~dp0
start "F1 Pit Wall Server" "D:\Anaconda3\envs\F1_graduation_project\python.exe" server.py --open
echo 服务启动中,浏览器将自动打开 http://127.0.0.1:8765/
echo 关闭本窗口不影响服务;停止服务请关闭 "F1 Pit Wall Server" 窗口。
timeout /t 3 >nul