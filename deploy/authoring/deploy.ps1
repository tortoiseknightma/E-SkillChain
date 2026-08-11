$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
& python (Join-Path $root "scripts\deploy_authoring_runtime.py") @args
if ($LASTEXITCODE -ne 0) {
    throw "Formal authoring runtime deployment failed with exit code $LASTEXITCODE"
}
