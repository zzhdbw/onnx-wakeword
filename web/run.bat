@echo off
echo Voicute Wake Word Demo
echo Open http://localhost:8080/web/
echo Press Ctrl+C to stop
cd /d "%~dp0.."
python -m http.server 8080
