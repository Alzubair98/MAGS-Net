param([string]$Python = "", [ValidateRange(1024, 65535)][int]$Port = 8501)
$ErrorActionPreference = 'Stop'
if (-not $Python) {
    $demoPython = Join-Path $PSScriptRoot '.venv/Scripts/python.exe'
    $Python = if (Test-Path -LiteralPath $demoPython) { $demoPython } else { (Get-Command python -ErrorAction Stop).Source }
}
Push-Location $PSScriptRoot
try {
    & $Python -m streamlit run magsnet_demo/app.py --server.port $Port
    if ($LASTEXITCODE -ne 0) { throw 'Streamlit exited with an error. Check the installed dependencies.' }
} finally { Pop-Location }
