Write-Host "Setting up environment for Lethal-Neuro-Blaze..."

# 1. Create venv if it doesn't exist
if (-not (Test-Path "venv")) {
    Write-Host "Creating virtual environment 'venv'..."
    python -m venv venv
} else {
    Write-Host "Virtual environment 'venv' already exists."
}

# 2. Activate venv (for this script's session)
Write-Host "Activating virtual environment..."
$venvPath = Join-Path $PSScriptRoot "venv\Scripts\Activate.ps1"
& $venvPath

# Verification check
if ($env:VIRTUAL_ENV) {
    Write-Host "Virtual environment activated: $env:VIRTUAL_ENV"
} else {
    Write-Host "WARNING: Virtual environment activation might have failed. Using direct paths for installation."
}

Write-Host "Upgrading pip..."
pip install --upgrade pip

# 3. Install dependencies
Write-Host "Installing opencv-python, pydirectinput, mss..."
pip install opencv-python pydirectinput mss

Write-Host "Installing pytorch (cuda 12.8)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

Write-Host " "
Write-Host "Setup complete!"
Write-Host "To activate the environment in your current shell, run:"
Write-Host ". .\venv\Scripts\Activate.ps1"
