@echo off
echo ========================================================
echo   Packaging Stock Prediction Project for Transfer
echo ========================================================
echo.
echo Cleaning up temporary python files...
REM Optionally remove pycache folders if they exist
FOR /d /r . %%d in (__pycache__) DO @IF EXIST "%%d" rd /s /q "%%d"

echo.
echo Creating ZIP archive (excluding 'venv' and hidden files)...
echo This might take a few seconds depending on data size...

REM Use PowerShell to zip the contents, excluding the virtual environment folder
powershell.exe -Command "Get-ChildItem -Path . -Exclude 'venv', '.git', '*.zip' | Compress-Archive -DestinationPath '..\stock_prediction_transfer.zip' -Force"

echo.
IF %ERRORLEVEL% EQU 0 (
    echo ========================================================
    echo SUCCESS! Your project has been zipped.
    echo You can find the file here:
    echo --^> %~dp0..\stock_prediction_transfer.zip
    echo ========================================================
) ELSE (
    echo [ERROR] Failed to create the ZIP file.
)

echo.
pause
