# ZT-Shield GitHub Deployment Script
# This script initializes a git repository (if needed) and pushes the project to GitHub.

$projectName = "adversarial-ml-security-framework"

Write-Host "--- ZT-Shield GitHub Deployment Automation ---" -ForegroundColor Cyan

# 1. Check if Git is installed
if (!(Get-Command git -ErrorAction SilentlyContinue)) {
    Write-Host "Error: Git is not installed. Please install Git from https://git-scm.com/" -ForegroundColor Red
    exit
}

# 2. Initialize Git if not already initialized
if (!(Test-Path .git)) {
    Write-Host "Initializing Git repository..." -ForegroundColor Yellow
    git init
    git branch -M main
}
else {
    Write-Host "Git repository already initialized." -ForegroundColor Green
}

# 3. Prompt for GitHub Information
$githubUser = Read-Host "Enter your GitHub username"
$repoName = Read-Host "Enter your GitHub repository name (default: $projectName)"
if ([string]::IsNullOrWhiteSpace($repoName)) { $repoName = $projectName }

$remoteUrl = "https://github.com/$githubUser/$repoName.git"

# 4. Stage and Commit
Write-Host "Staging files..." -ForegroundColor Yellow
git add .

Write-Host "Creating initial commit..." -ForegroundColor Yellow
git commit -m "Initial Research Release: ZT-Shield v1.0"

# 5. Handle Remote
$existingRemote = git remote get-url origin 2>$null
if ($null -ne $existingRemote) {
    if ($existingRemote -ne $remoteUrl) {
        Write-Host "Updating existing remote URL..." -ForegroundColor Yellow
        git remote set-url origin $remoteUrl
    }
}
else {
    Write-Host "Adding GitHub remote..." -ForegroundColor Yellow
    git remote add origin $remoteUrl
}

# 6. Push to GitHub
Write-Host "`n--- Ready to push ---" -ForegroundColor Cyan
Write-Host "Target: $remoteUrl"
$confirm = Read-Host "Do you want to push to GitHub now? (y/n)"

if ($confirm -eq 'y') {
    Write-Host "Pushing to GitHub (main branch)..." -ForegroundColor Yellow
    git push -u origin main
    
    if ($LASTEXITCODE -eq 0) {
        Write-Host "`nSuccess! Your project is now on GitHub." -ForegroundColor Green
        Write-Host "View it at: https://github.com/$githubUser/$repoName" -ForegroundColor Cyan
    }
    else {
        Write-Host "`nPush failed. Ensure you have created the repository on GitHub first." -ForegroundColor Red
        Write-Host "Create it here: https://github.com/new" -ForegroundColor Cyan
    }
}
else {
    Write-Host "Push cancelled. You can push manually using 'git push -u origin main'." -ForegroundColor Gray
}

Write-Host "`nPress any key to exit..."
$null = [Console]::ReadKey($true)
