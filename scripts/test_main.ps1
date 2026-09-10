param(
    [string]$Python = "python",
    [switch]$OverallOnly
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$ProjectRoot = Resolve-Path (Join-Path $ScriptDir "..")
Set-Location $ProjectRoot

$Args = @("evaluate_main.py")
if ($OverallOnly) { $Args += "--no-print-per-snr" }
& $Python @Args
if ($LASTEXITCODE -ne 0) { throw "Main evaluation failed" }
