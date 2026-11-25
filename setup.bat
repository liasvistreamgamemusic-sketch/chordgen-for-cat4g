@echo off
rem Run setup only (first-time preparation, retry, or switching GPU). Same as "run.bat --setup".
rem Examples: setup.bat / setup.bat --gpu nvidia / setup.bat --gpu amd --amd-device gfx1201
rem           setup.bat --rocm 7.2.1 / setup.bat --cuda 12.6 / setup.bat --force
call "%~dp0run.bat" --setup %*
exit /b %ERRORLEVEL%
