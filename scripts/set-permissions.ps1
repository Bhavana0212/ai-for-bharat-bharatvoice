<<<<<<< HEAD
# PowerShell script to set executable permissions on shell scripts
# This is for Windows environments where chmod is not available

Write-Host "Setting executable permissions on shell scripts..." -ForegroundColor Green

$scriptFiles = Get-ChildItem -Path "scripts" -Filter "*.sh"

foreach ($file in $scriptFiles) {
    Write-Host "Processing: $($file.Name)" -ForegroundColor Yellow
    
    # On Windows, we can't set Unix permissions, but we can ensure the files are not read-only
    if ($file.IsReadOnly) {
        $file.IsReadOnly = $false
        Write-Host "  Removed read-only attribute" -ForegroundColor Cyan
    }
    
    # Add shebang if not present
    $content = Get-Content $file.FullName -Raw
    if (-not $content.StartsWith("#!/bin/bash")) {
        Write-Host "  Note: Ensure script starts with #!/bin/bash for Unix execution" -ForegroundColor Yellow
    }
}

Write-Host "Permission setup completed!" -ForegroundColor Green
Write-Host ""
=======
# PowerShell script to set executable permissions on shell scripts
# This is for Windows environments where chmod is not available

Write-Host "Setting executable permissions on shell scripts..." -ForegroundColor Green

$scriptFiles = Get-ChildItem -Path "scripts" -Filter "*.sh"

foreach ($file in $scriptFiles) {
    Write-Host "Processing: $($file.Name)" -ForegroundColor Yellow
    
    # On Windows, we can't set Unix permissions, but we can ensure the files are not read-only
    if ($file.IsReadOnly) {
        $file.IsReadOnly = $false
        Write-Host "  Removed read-only attribute" -ForegroundColor Cyan
    }
    
    # Add shebang if not present
    $content = Get-Content $file.FullName -Raw
    if (-not $content.StartsWith("#!/bin/bash")) {
        Write-Host "  Note: Ensure script starts with #!/bin/bash for Unix execution" -ForegroundColor Yellow
    }
}

Write-Host "Permission setup completed!" -ForegroundColor Green
Write-Host ""
>>>>>>> 0eb0e95caee35c9eb86ecf88b155e812550321aa
Write-Host "Note: On Unix/Linux systems, run: chmod +x scripts/*.sh" -ForegroundColor Cyan