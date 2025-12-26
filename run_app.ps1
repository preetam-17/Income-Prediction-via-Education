Write-Host "Income Prediction App Setup" -ForegroundColor Green
Write-Host "=========================" -ForegroundColor Green
Write-Host

# Check if Python is installed
try {
    $pythonVersion = python --version
    Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
    $pythonCmd = "python"
} catch {
    try {
        $pythonVersion = py --version
        Write-Host "Python detected: $pythonVersion" -ForegroundColor Green
        $pythonCmd = "py"
    } catch {
        Write-Host "Python not found. Please install Python from python.org" -ForegroundColor Red
        Write-Host "Press any key to exit..."
        $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
        exit
    }
}

# Check for required packages
Write-Host "Checking for required packages..." -ForegroundColor Yellow
& $pythonCmd -m pip install -r requirements.txt
if ($LASTEXITCODE -ne 0) {
    Write-Host "Failed to install required packages. Please check your internet connection." -ForegroundColor Red
    Write-Host "Press any key to exit..."
    $null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")
    exit
}

# Train the model
Write-Host "Training the income prediction model..." -ForegroundColor Yellow
& $pythonCmd train_model.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Model training failed, but we'll continue with simulated predictions." -ForegroundColor Yellow
} else {
    Write-Host "Model training complete!" -ForegroundColor Green
}

# Start the Flask app
Write-Host "Starting the Flask application..." -ForegroundColor Yellow
Write-Host "Once started, open your browser and go to: http://127.0.0.1:5000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server when done." -ForegroundColor Cyan
Write-Host

& $pythonCmd -m flask run 