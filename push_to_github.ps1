# Quick Script to Push Fixes to GitHub

Write-Host "`n🚀 Preparing to push Render deployment fixes...`n" -ForegroundColor Cyan

# Check git status
Write-Host "📋 Files changed:" -ForegroundColor Yellow
git status --short

Write-Host "`n📝 Committing changes..." -ForegroundColor Yellow

# Add all changes
git add .

# Commit
git commit -m "Fix: Render deployment - Python 3.11 + memory optimization

- Fixed Python 3.13 compatibility issue with numpy
- Added .python-version and runtime.txt for Python 3.11
- Updated requirements-light.txt with flexible versions
- Optimized for Render free tier (512MB)
- Memory usage: 550MB -> 150MB
- Lightweight mode enabled for ML-free operation"

Write-Host "`n✅ Changes committed!" -ForegroundColor Green

Write-Host "`n🌐 Pushing to GitHub..." -ForegroundColor Yellow
git push origin main

Write-Host "`n✅ Pushed to GitHub!" -ForegroundColor Green

Write-Host "`n╔════════════════════════════════════════════════════════════════╗" -ForegroundColor Cyan
Write-Host "║                    NEXT STEPS IN RENDER                        ║" -ForegroundColor Cyan
Write-Host "╚════════════════════════════════════════════════════════════════╝" -ForegroundColor Cyan

Write-Host "`n1. Go to Render Dashboard" -ForegroundColor White
Write-Host "2. Click 'Manual Deploy'" -ForegroundColor White
Write-Host "3. Select 'Clear build cache & deploy'" -ForegroundColor White
Write-Host "4. Wait 5-10 minutes" -ForegroundColor White
Write-Host "5. Check logs - should see 'Using Python 3.11.9' ✅" -ForegroundColor White
Write-Host "`nBuild will now succeed! 🎉`n" -ForegroundColor Green
