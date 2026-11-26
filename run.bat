@echo off
setlocal EnableExtensions DisableDelayedExpansion
rem chordgen portable launcher. Keep this file ASCII-only: cmd.exe reads .bat files in the
rem console code page (932 on Japanese Windows), so UTF-8 text here can break parsing.
rem Details (in Japanese) are in README.md.
rem
rem   run.bat                                   start the browser GUI (python -m chordgen.gui)
rem   run.bat --gui [--port N] [--no-browser]   start the GUI with options
rem   run.bat <audio file or URL> [options]     analyze on the command line (run.bat --help for help)
rem   run.bat --setup [--gpu amd|nvidia|cpu]    run setup only (same as setup.bat)
rem   run.bat --media-dl <args>                 call the bundled media-dl (e.g. update)
rem   run.bat --python <args>                   call the bundled Python (e.g. -m chordgen.eval --help)
rem
rem Every variable below points into this folder. setlocal keeps them inside this file,
rem so Windows user/system settings are never changed.

rem ---- app folder (where this file is), without the trailing backslash ----
set "CG_APP=%~dp0"
set "CHORDGEN_HOME=%CG_APP:~0,-1%"
set "CG_RUNTIME=%CHORDGEN_HOME%\runtime"
set "CG_CACHE=%CHORDGEN_HOME%\cache"
set "CG_MODELS=%CHORDGEN_HOME%\models"
set "CG_WORK=%CHORDGEN_HOME%\work"

rem ---- variables required by the spec (see README) ----
rem yt-dlp, ffmpeg, ffprobe, deno for media-dl (the analysis side shares this ffmpeg)
set "MEDIA_DL_BIN_DIR=%CG_RUNTIME%\media-dl\bin"
rem caches such as yt-dlp (some tools expand ~ to HOME on Windows)
set "XDG_CACHE_HOME=%CG_CACHE%\xdg"
set "HOME=%CG_CACHE%\home"
rem config files
set "XDG_CONFIG_HOME=%CG_CACHE%\config"
rem MIOpen tuning db and kernel cache
set "MIOPEN_USER_DB_PATH=%CG_CACHE%\miopen\db"
set "MIOPEN_CUSTOM_CACHE_DIR=%CG_CACHE%\miopen\kernels"
rem MIOpen: errors only (its warnings about tuning files are harmless)
set "MIOPEN_LOG_LEVEL=3"
rem model weights
set "HF_HOME=%CG_MODELS%\huggingface"
rem Hugging Face: no symlinks on Windows without Developer Mode (works, just warns); no telemetry
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"
set "HF_HUB_DISABLE_TELEMETRY=1"
set "TORCH_HOME=%CG_MODELS%\torch"
rem Triton, pip and deno caches
set "TRITON_CACHE_DIR=%CG_CACHE%\triton"
set "PIP_CACHE_DIR=%CG_CACHE%\pip"
set "DENO_DIR=%CG_CACHE%\deno"
rem temporary files
set "TEMP=%CG_WORK%\tmp"
set "TMP=%CG_WORK%\tmp"
rem do not load user site-packages
set "PYTHONNOUSERSITE=1"

rem ---- not in the spec table, but these default to LOCALAPPDATA / APPDATA / the user profile ----
set "npm_config_cache=%CG_CACHE%\npm"
set "npm_config_prefix=%CG_RUNTIME%\npm-global"
set "npm_config_userconfig=%CG_CACHE%\config\npmrc"
set "npm_config_update_notifier=false"
set "npm_config_fund=false"
set "npm_config_audit=false"
set "CUDA_CACHE_PATH=%CG_CACHE%\nvidia\ComputeCache"
set "MPLCONFIGDIR=%CG_CACHE%\matplotlib"
set "NUMBA_CACHE_DIR=%CG_CACHE%\numba"
set "TORCH_EXTENSIONS_DIR=%CG_CACHE%\torch_extensions"
set "TORCHINDUCTOR_CACHE_DIR=%CG_CACHE%\torchinductor"

rem ---- isolate Python from Windows-side settings ----
rem a PYTHONHOME set on Windows would make the bundled Python load another stdlib
set "PYTHONHOME="
set "PYTHONSTARTUP="
set "VIRTUAL_ENV="
rem chordgen is imported from this folder, not installed (the folder stays movable)
set "PYTHONPATH=%CHORDGEN_HOME%"
set "PYTHONUTF8=1"
rem ignore the user's pip.ini and any require-virtualenv setting
set "PIP_CONFIG_FILE=nul"
set "PIP_REQUIRE_VIRTUALENV=false"
set "PIP_DISABLE_PIP_VERSION_CHECK=1"

rem ---- bundled tools ----
set "CHORDGEN_PYTHON=%CG_RUNTIME%\python\python.exe"
set "CHORDGEN_NODE=%CG_RUNTIME%\node\node.exe"
set "CG_MEDIA_DL_JS=%CHORDGEN_HOME%\node_modules\@liasvistreamgamemusic-sketch\media-dl\dist\cli.js"
rem put bundled Python, Node.js and MEDIA_DL_BIN_DIR first (ffmpeg is shared by media-dl and analysis)
set "PATH=%CG_RUNTIME%\python;%CG_RUNTIME%\python\Scripts;%CG_RUNTIME%\node;%MEDIA_DL_BIN_DIR%;%PATH%"

rem create output folders up front (some libraries fail if they do not exist)
for %%d in ("%TEMP%" "%HOME%" "%XDG_CACHE_HOME%" "%XDG_CONFIG_HOME%" "%MIOPEN_USER_DB_PATH%" "%MIOPEN_CUSTOM_CACHE_DIR%" "%HF_HOME%" "%TORCH_HOME%" "%TRITON_CACHE_DIR%" "%PIP_CACHE_DIR%" "%DENO_DIR%" "%MEDIA_DL_BIN_DIR%") do if not exist "%%~d\" mkdir "%%~d" 2>nul

set "CG_RC=0"
if /i "%~1"=="--setup" goto setup_explicit
if /i "%~1"=="--media-dl" goto media_dl
if /i "%~1"=="--python" goto python
if not exist "%CHORDGEN_PYTHON%" goto setup_auto
if not exist "%CG_RUNTIME%\setup-complete.json" goto setup_auto
goto run

:setup_auto
echo [chordgen] Setup is not complete. Running setup first (internet connection required) ...
call :bootstrap
if errorlevel 1 goto fail
"%CHORDGEN_PYTHON%" -P -s "%CHORDGEN_HOME%\portable\install.py"
if errorlevel 1 goto fail
goto run

:setup_explicit
call :bootstrap
if errorlevel 1 goto fail
rem install.py ignores the leading --setup
"%CHORDGEN_PYTHON%" -P -s "%CHORDGEN_HOME%\portable\install.py" %*
set "CG_RC=%ERRORLEVEL%"
goto end

:media_dl
if not exist "%CG_MEDIA_DL_JS%" goto media_dl_missing
set "CG_NEXT=media_dl_call"
goto collect_args
:media_dl_call
"%CHORDGEN_NODE%" "%CG_MEDIA_DL_JS%"%CG_ARGS%
set "CG_RC=%ERRORLEVEL%"
goto end

:python
if not exist "%CHORDGEN_PYTHON%" goto setup_auto_python
set "CG_NEXT=python_call"
goto collect_args
:python_call
"%CHORDGEN_PYTHON%" -P -s%CG_ARGS%
set "CG_RC=%ERRORLEVEL%"
goto end

:setup_auto_python
echo [chordgen] Python is not installed yet. Run setup.bat first.
set "CG_RC=1"
goto end

rem collect the arguments after the first one into CG_ARGS, then goto CG_NEXT.
rem (shift does not change %%*; "call :label %%*" is avoided because call expands %% twice,
rem which would break URLs such as ...%%20...)
:collect_args
set "CG_ARGS="
shift
:collect_args_loop
if "%~1"=="" goto %CG_NEXT%
set CG_ARGS=%CG_ARGS% %1
shift
goto collect_args_loop

:media_dl_missing
echo [chordgen] media-dl is not installed. Run setup.bat (see README).
set "CG_RC=1"
goto end

:run
if "%~1"=="" goto gui
if /i "%~1"=="--gui" goto gui_opts
goto run_args

rem no arguments (e.g. double-clicked): start the GUI. On failure keep the window open
rem so the message can be read.
:gui
"%CHORDGEN_PYTHON%" -P -s -m chordgen.gui
set "CG_RC=%ERRORLEVEL%"
if not "%CG_RC%"=="0" pause
goto end

:gui_opts
set "CG_NEXT=gui_call"
goto collect_args
:gui_call
"%CHORDGEN_PYTHON%" -P -s -m chordgen.gui%CG_ARGS%
set "CG_RC=%ERRORLEVEL%"
goto end

:run_args
"%CHORDGEN_PYTHON%" -P -s -m chordgen.cli %*
set "CG_RC=%ERRORLEVEL%"
goto end

:bootstrap
"%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe" -NoProfile -NonInteractive -ExecutionPolicy Bypass -File "%CHORDGEN_HOME%\portable\bootstrap.ps1"
exit /b %ERRORLEVEL%

:fail
echo.
echo [chordgen] Setup failed. Fix the problem shown above, then run setup.bat again (it resumes where it stopped).
set "CG_RC=1"
goto end

:end
endlocal & exit /b %CG_RC%
