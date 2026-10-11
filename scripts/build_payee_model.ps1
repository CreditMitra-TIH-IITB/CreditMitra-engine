# Build the Ollama payee model from a merged Hugging Face checkpoint.
#
#   powershell -ExecutionPolicy Bypass -File scripts\build_payee_model.ps1 `
#       -MergedDir ..\outputs\payee-merged -LlamaCpp ..\outputs\llama.cpp
#
# MergedDir is the base model with a payee adapter merged in (see
# CreditMitra-finetune). To serve a different adapter, merge it and pass that
# directory instead - nothing else changes.
#
# Quantized to Q8_0: half the size of F16 with the same accuracy on the payee
# validation set (82.7% exact match on 150 samples, both builds) and ~2.4x
# faster per call. Ollama's own create-time quantization does not support the
# Qwen2 architecture, so conversion goes through llama.cpp's converter.
param(
    [Parameter(Mandatory = $true)][string]$MergedDir,
    [Parameter(Mandatory = $true)][string]$LlamaCpp,
    [string]$Tag = "payee-lora:latest",
    [string]$OutType = "q8_0"
)
$ErrorActionPreference = "Stop"
$MergedDir = (Resolve-Path $MergedDir).Path
$LlamaCpp = (Resolve-Path $LlamaCpp).Path
$Gguf = Join-Path (Split-Path $MergedDir -Parent) ("payee-" + $OutType + ".gguf")

Write-Host "Converting $MergedDir -> $Gguf ($OutType)..."
python (Join-Path $LlamaCpp "convert_hf_to_gguf.py") $MergedDir --outtype $OutType --outfile $Gguf
if ($LASTEXITCODE -ne 0) { throw "GGUF conversion failed (pip install gguf sentencepiece?)" }

$Modelfile = Join-Path $env:TEMP "payee.Modelfile"
@"
FROM $($Gguf -replace '\\', '/')

PARAMETER temperature 0
PARAMETER num_ctx 4096

TEMPLATE """{{ if .System }}<|im_start|>system
{{ .System }}
{{ end }}{{ if .Prompt }}<|im_start|>user
{{ .Prompt }}
{{ end }}<|im_start|>assistant
"""
"@ | Set-Content -Path $Modelfile -Encoding ascii

ollama create $Tag -f $Modelfile
if ($LASTEXITCODE -ne 0) { throw "ollama create failed" }
Write-Host "Built $Tag from $MergedDir"
