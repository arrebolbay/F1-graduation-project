@echo off
chcp 65001 >nul
REM ============================================================
REM  一键提交并推送 F1毕设 仓库到 GitHub
REM  https://github.com/arrebolbay/F1-graduation-project
REM  用法: 双击运行;或命令行  推送更新.bat "提交说明"
REM ============================================================
cd /d "%~dp0"

git add -A

git diff --cached --quiet
if %errorlevel%==0 (
    echo [提示] 没有变更需要推送。
    goto :end
)

set "MSG=%~1"
if "%MSG%"=="" set "MSG=update: 自动同步 %date% %time%"

git commit -m "%MSG%"
if errorlevel 1 (
    echo [错误] 提交失败,请检查上方输出。
    goto :end
)

git push origin main
if errorlevel 1 (
    echo [错误] 推送失败 —— 多为网络原因,稍后重新双击本脚本即可续传。
    goto :end
)

echo [完成] 已推送到 GitHub。

:end
pause
