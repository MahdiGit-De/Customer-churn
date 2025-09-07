param(
  [Parameter(Mandatory=$true)][string]$Owner,
  [Parameter(Mandatory=$true)][string]$Repo,
  [Parameter(Mandatory=$true)][string]$Description,
  [string[]]$Topics
)

# Usage:
#   $env:GITHUB_TOKEN = '<YOUR_PAT_WITH_REPO_SCOPE>'
#   .\scripts\set_repo_metadata.ps1 -Owner 'MahdiGit-De' -Repo 'Customer-churn' -Description 'Customer churn prediction with LightGBM & SHAP' -Topics 'ml','lightgbm','shap','churn','telco'

if (-not $env:GITHUB_TOKEN) {
  Write-Error 'GITHUB_TOKEN env var not set. Create a PAT with repo scope and set $env:GITHUB_TOKEN.'
  exit 1
}

$Headers = @{ 
  Authorization = "Bearer $($env:GITHUB_TOKEN)"
  Accept        = 'application/vnd.github+json'
  'X-GitHub-Api-Version' = '2022-11-28'
  'User-Agent' = 'codex-cli'
}

# Update description
$repoUrl = "https://api.github.com/repos/$Owner/$Repo"
$body = @{ description = $Description } | ConvertTo-Json
Invoke-RestMethod -Method Patch -Uri $repoUrl -Headers $Headers -Body $body | Out-Null
Write-Output "Updated description for $Owner/$Repo"

if ($Topics -and $Topics.Length -gt 0) {
  $topicsUrl = "https://api.github.com/repos/$Owner/$Repo/topics"
  $topicsBody = @{ names = $Topics } | ConvertTo-Json
  Invoke-RestMethod -Method Put -Uri $topicsUrl -Headers ($Headers + @{ 'Accept' = 'application/vnd.github.mercy-preview+json' }) -Body $topicsBody | Out-Null
  Write-Output "Set topics: $($Topics -join ', ')"
}

