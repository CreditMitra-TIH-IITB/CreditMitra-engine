# Start the CreditMitra stack locally: Ollama, the engine, and the dashboard.
#
#   powershell -ExecutionPolicy Bypass -File scripts\dev.ps1
#
# Expects the dashboard checked out next to the engine (..\CreditMitra-dashboard).
# See docs/RUNNING.md for one-time setup.
param(
    [string]$Dashboard = (Join-Path $PSScriptRoot "..\..\CreditMitra-dashboard"),
    [int]$EnginePort = 8000
)
$ErrorActionPreference = "Stop"
$Engine = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$OllamaUrl = "http://127.0.0.1:11434"

function Test-Url([string]$Url) {
    try { Invoke-RestMethod $Url -TimeoutSec 3 | Out-Null; return $true } catch { return $false }
}

# 1. Ollama -------------------------------------------------------------------
if (-not (Test-Url "$OllamaUrl/api/tags")) {
    Write-Host "Starting Ollama..."
    Start-Process ollama -ArgumentList "serve" -WindowStyle Hidden
    for ($i = 0; $i -lt 20 -and -not (Test-Url "$OllamaUrl/api/tags"); $i++) { Start-Sleep 1 }
    if (-not (Test-Url "$OllamaUrl/api/tags")) { throw "Ollama did not start. Is it installed?" }
}
$models = (Invoke-RestMethod "$OllamaUrl/api/tags").models.name
if ($models -notcontains "payee-lora:latest") {
    Write-Warning "Ollama has no 'payee-lora' model - every payee will come back empty. Build it with scripts\build_payee_model.ps1."
}

# 2. Classifier model files ---------------------------------------------------
$required = @(
    "data\models\attentionmlp.onnx",
    "data\models\scaler.joblib",
    "data\models\onnx_pipeline\qwen3_embedding.onnx"
)
$missing = $required | Where-Object { -not (Test-Path (Join-Path $Engine $_)) }
if ($missing) {
    Write-Warning "Missing classifier files ($($missing -join ', ')). Payees will be classified by keyword heuristics. See data\models\README.md."
}

# 3. Engine -------------------------------------------------------------------
$Python = Join-Path $Engine "venv\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    throw "No venv. Run: python -m venv venv; venv\Scripts\pip install -e "".[dev]"""
}
Write-Host "Starting engine on port $EnginePort (new window)..."
Start-Process $Python -ArgumentList "-m", "uvicorn", "app.main:app", "--port", $EnginePort -WorkingDirectory $Engine

# 4. Dashboard ----------------------------------------------------------------
if (Test-Path $Dashboard) {
    $Dashboard = (Resolve-Path $Dashboard).Path
    if (-not (Test-Path (Join-Path $Dashboard "node_modules"))) {
        Write-Host "Installing dashboard dependencies..."
        Push-Location $Dashboard
        try { npm install } finally { Pop-Location }
    }
    Write-Host "Starting dashboard (new window)..."
    Start-Process powershell -ArgumentList "-NoExit", "-Command", "npm run dev" -WorkingDirectory $Dashboard
} else {
    Write-Warning "Dashboard not found at $Dashboard - start it yourself with 'npm run dev'."
}

# 5. Wait for the engine ------------------------------------------------------
for ($i = 0; $i -lt 60 -and -not (Test-Url "http://127.0.0.1:$EnginePort/health"); $i++) { Start-Sleep 1 }
if (Test-Url "http://127.0.0.1:$EnginePort/health") {
    Write-Host ""
    Write-Host "Engine:    http://127.0.0.1:$EnginePort/docs"
    Write-Host "Dashboard: http://localhost:5173"
    Write-Host "Models keep loading in the background for ~1-2 minutes after start."
} else {
    Write-Warning "Engine did not answer /health within 60s - check its window for errors."
}
