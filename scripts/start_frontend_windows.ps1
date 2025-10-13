Param(
  [int]$Port = 3000,
  [switch]$Open
)

$ErrorActionPreference = "Stop"

# Resolve project root and frontend directory
$ROOT = (Resolve-Path "$PSScriptRoot\..").Path
$FE = Join-Path $ROOT "frontend"

Write-Host "Project root: $ROOT"
Write-Host "Frontend dir: $FE"

if (-not (Test-Path $FE)) {
  Write-Error "Frontend directory not found at $FE"
  exit 1
}

# Ensure Node and npm are available
try {
  $nodeVer = node --version
  Write-Host "Node detected: $nodeVer"
} catch {
  Write-Error "Node.js is not available in PATH. Please install Node 18+ and retry."
  exit 1
}
try {
  $npmVer = npm --version
  Write-Host "npm detected: $npmVer"
} catch {
  Write-Error "npm is not available in PATH. Please install Node/npm and retry."
  exit 1
}

Push-Location $FE
try {
  Write-Host "Installing frontend dependencies"
  npm install

  # Vite port is configured in vite.config.js to 3000; optionally override
  if ($Port -ne 3000) {
    Write-Host "Note: vite.config.js sets port=3000; update config to change port."
  }

  Write-Host "Starting frontend dev server: npm run dev"
  npm run dev
} finally {
  Pop-Location
}