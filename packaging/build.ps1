# Build dist\RawCellInspection.exe from the repo root:
#     .\packaging\build.ps1
# Anaconda folders are dropped from PATH first: PyInstaller otherwise bundles
# Anaconda's Qt / MSVC runtime DLLs, and the exe fails with "DLL load failed
# while importing QtCore".
$root = Split-Path $PSScriptRoot -Parent
$env:Path = ($env:Path -split ';' | Where-Object { $_ -notmatch 'anaconda|miniconda|conda' }) -join ';'
& "$root\venv_rci\Scripts\pyinstaller.exe" "$root\packaging\RawCellInspection.spec" --noconfirm `
    --distpath "$root\dist" --workpath "$root\build"
if ($LASTEXITCODE -ne 0) { throw "PyInstaller failed with exit code $LASTEXITCODE" }
