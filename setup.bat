@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo === [1/4] 创建虚拟环境 ===
py -3.11 -m venv .venv
echo === [2/4] 激活虚拟环境 ===
call .venv\Scripts\activate.bat
echo === [3/4] 升级 pip ===
python -m pip install --upgrade pip
echo === [4/4] 安装依赖（使用清华镜像加速，下载较大，请耐心等待） ===
python -m pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
echo.
echo 安装完成！以后启动网页请双击 run_app.bat
pause
