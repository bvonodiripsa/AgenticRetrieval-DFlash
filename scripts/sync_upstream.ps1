#!/usr/bin/env pwsh
# =============================================================================
# Sync the upstream AgenticRetrieval repo into external/agenticretrieval.
#
# This project uses the upstream source AS-IS (the folder is git-ignored), so it
# can be re-synced at any time without merge conflicts against local forks.
#
# Usage:
#   ./scripts/sync_upstream.ps1                 # clone or fast-forward pull
#   ./scripts/sync_upstream.ps1 -Ref v1.2.3     # check out a specific ref/tag
# =============================================================================
param(
    [string]$Ref = ""
)
$ErrorActionPreference = "Stop"

$root = Split-Path -Parent $PSScriptRoot
$dest = Join-Path $root "external/agenticretrieval"
$url  = "https://github.com/AzureCosmosDB/AgenticRetrieval.git"

if (Test-Path (Join-Path $dest ".git")) {
    Write-Host "Updating upstream in $dest ..."
    git -C $dest fetch --all --prune
    if ($Ref) { git -C $dest checkout $Ref } else { git -C $dest pull --ff-only }
} else {
    Write-Host "Cloning $url into $dest ..."
    New-Item -ItemType Directory -Force -Path (Split-Path $dest) | Out-Null
    git clone $url $dest
    if ($Ref) { git -C $dest checkout $Ref }
}

Write-Host "Upstream synced at: $dest"
