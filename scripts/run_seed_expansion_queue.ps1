# Print or run the seed-expansion queue (Windows helper).
# Usage:
#   .\scripts\run_seed_expansion_queue.ps1 -DryRun
#   .\scripts\run_seed_expansion_queue.ps1 -Only p0 -DryRun
#   .\scripts\run_seed_expansion_queue.ps1 -Only p0 -Execute   # GPU

param(
    [ValidateSet("p0", "p1", "")]
    [string]$Only = "",
    [switch]$DryRun,
    [switch]$Execute
)

$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root

$argsList = @("scripts/run_seed_expansion_queue.py")
if ($Only) { $argsList += @("--only", $Only) }
if ($Execute) { $argsList += "--execute" } else { $argsList += "--dry-run" }

$py = Join-Path $Root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { $py = "python" }
& $py @argsList
exit $LASTEXITCODE
