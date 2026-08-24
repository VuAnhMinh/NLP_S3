# Overnight, unattended re-run of the S3 portion of the CafeBERT benchmark,
# using refit_transform() + estimate_components() instead of one independent
# .fit() per (variant, seed, k). Goal: measure the real speedup from avoiding
# redundant CafeBERT vocabulary re-encoding (see run_cafebert_refit_optimized.py
# docstring for the full root-cause explanation).
#
# Does NOT touch benchmark/cafebert_full/reference/ (the committed, audited
# 480-row artifact) -- writes to benchmark/cafebert_full/results_refit/.
#
# Usage (from repo root, e.g. E:\Development\NLP_S3):
#   powershell -ExecutionPolicy Bypass -File benchmark\cafebert_full\run_overnight_refit_experiment.ps1
#
# Safe to just run and go to sleep: every step is wrapped so a failure is
# logged (with the real exit code -- native command failures are checked
# explicitly, since PowerShell does NOT turn a non-zero exe exit code into a
# catchable exception on its own) and does not silently pretend to succeed.
# The script resumes from where it left off if you run it again (both the
# fetch steps and run_cafebert_refit_optimized.py skip already-completed work).

$ErrorActionPreference = "Continue"
$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $RepoRoot

$LogDir = Join-Path $PSScriptRoot "overnight_logs"
New-Item -ItemType Directory -Force -Path $LogDir | Out-Null
$Stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$LogFile = Join-Path $LogDir "run_$Stamp.log"

function Write-Log {
    param([string]$Message)
    $line = "[{0}] {1}" -f (Get-Date -Format "HH:mm:ss"), $Message
    Write-Host $line
    Add-Content -Path $LogFile -Value $line
}

# Runs one native command (python.exe, py.exe, ...) and THROWS if it exits
# non-zero. PowerShell does not do this on its own for native executables --
# a failed python.exe call silently falls through to the next line unless you
# check $LASTEXITCODE yourself. Every call to an external exe in this script
# must go through this function, or a failure will be missed exactly like the
# first version of this script missed every failure tonight.
function Invoke-Native {
    # IMPORTANT: emits nothing to the success/output stream on purpose. Any
    # unsuppressed output here becomes part of THIS function's return value,
    # which then contaminates whatever variable the caller assigns from
    # Invoke-Step below -- a non-empty array is always "truthy" in PowerShell
    # regardless of whether the real result was $true or $false, so
    # `if (-not $SomeResult)` silently stops working. Wrote to console via
    # Write-Host (bypasses the success stream) and to the log file via
    # Add-Content instead of Tee-Object, specifically to avoid that trap --
    # it's exactly what let the smoke-test failure fall through uncaught.
    param([Parameter(Mandatory = $true)][string]$Exe, [Parameter(ValueFromRemainingArguments = $true)]$ExtraArgs)
    & $Exe @ExtraArgs 2>&1 | ForEach-Object {
        Write-Host $_
        Add-Content -Path $LogFile -Value $_
    }
    if ($LASTEXITCODE -ne 0) {
        throw "$Exe $($ExtraArgs -join ' ') exited with code $LASTEXITCODE"
    }
}

function Invoke-Step {
    param([string]$Name, [scriptblock]$Body, [bool]$Fatal = $false)
    Write-Log "=== BEGIN: $Name ==="
    try {
        & $Body | Out-Null   # defensively swallow any incidental output from $Body
        Write-Log "=== OK: $Name ==="
        return $true
    } catch {
        Write-Log "=== FAILED: $Name -- $($_.Exception.Message) ==="
        if ($Fatal) {
            Write-Log "Fatal step failed, stopping."
            exit 1
        }
        return $false
    }
}

Write-Log "Repo root: $RepoRoot"
Write-Log "Log file: $LogFile"

# --- 1. Dedicated venv, pinned to Python 3.11/3.12/3.10 explicitly. ---
# scipy==1.13.1 (pinned in requirements.txt, matches the audited reference
# run) has no prebuilt Windows wheel for Python 3.13 -- pip falls back to
# building from source, which needs a Fortran compiler this machine doesn't
# have. Using plain "python -m venv" picks whatever "python" resolves to on
# PATH, which was 3.13 here. Pick an explicit, known-compatible version via
# the py launcher instead.
$VenvDir = Join-Path $RepoRoot ".venv-cafebert"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    $PickedTag = $null
    foreach ($tag in @("-3.11", "-3.12", "-3.10")) {
        $probe = & py $tag -c "print('ok')" 2>$null
        if ($LASTEXITCODE -eq 0 -and $probe -eq "ok") {
            $PickedTag = $tag
            break
        }
    }
    if (-not $PickedTag) {
        Write-Log "FATAL: no Python 3.10/3.11/3.12 found via the 'py' launcher."
        Write-Log "scipy==1.13.1 needs one of these (no Windows wheel for 3.13 without a Fortran compiler)."
        Write-Log "Install Python 3.11 from python.org, then re-run this script."
        exit 1
    }
    Write-Log "Creating .venv-cafebert with Python $PickedTag"
    Invoke-Step -Name "Create .venv-cafebert" -Fatal $true -Body {
        Invoke-Native py $PickedTag -m venv $VenvDir
    }
}

Invoke-Step -Name "Install benchmark/cafebert_full/requirements.txt" -Fatal $true -Body {
    Invoke-Native $VenvPython -m pip install --upgrade pip
    Invoke-Native $VenvPython -m pip install -r "benchmark\cafebert_full\requirements.txt"
}

# --- 2. unrar check (needed for VNTC-CNTT source archive only) ---
$UnrarFound = Get-Command unrar -ErrorAction SilentlyContinue
if (-not $UnrarFound) {
    Write-Log "WARNING: 'unrar' not found on PATH. VNTC-CNTT source fetch will likely fail;"
    Write-Log "the other three corpora (vietnamese-news, visfd, vi-medical) are unaffected."
    Write-Log "Install with: winget install RARLab.WinRAR   (adds unrar.exe to a WinRAR folder --"
    Write-Log "you may need to add that folder to PATH yourself), then re-run this script."
}

# --- 3. Fetch source snapshots + lock revisions (not timed, per README) ---
$SourcesOk = Invoke-Step -Name "fetch_sources" -Body {
    Invoke-Native $VenvPython -m benchmark.cafebert_full.fetch_sources
}

# --- 4. Fetch CafeBERT checkpoint ---
Invoke-Step -Name "fetch_cafebert_checkpoint" -Fatal $true -Body {
    Invoke-Native $VenvPython -m benchmark.cafebert_full.fetch_cafebert_checkpoint
}
$env:S3_CAFEBERT_CHECKPOINT_DIR = Join-Path $RepoRoot "benchmark\cafebert_full\pretrained\CafeBERT"

# --- 5. Smoke test first -- catches encoder/env problems on a small grid before the long run ---
$SmokeOk = Invoke-Step -Name "run_cafebert_smoke (sanity check)" -Body {
    Invoke-Native $VenvPython -m benchmark.cafebert_full.run_cafebert_smoke
}
if (-not $SmokeOk) {
    Write-Log "Smoke test failed -- stopping before the long run so this doesn't burn all night on a broken setup."
    Write-Log "Fix the error above, then re-run this script (it resumes)."
    exit 1
}

# --- 6. Seed results_refit/representation_cache/ from the already-committed reference
#        cache, so we reuse existing document embeddings instead of re-encoding them
#        (only fresh CafeBERT work needed is vocabulary encoding on the anchor fit). ---
$env:S3_CAFEBERT_RESULTS_DIR = Join-Path $RepoRoot "benchmark\cafebert_full\results_refit"
$NewCacheDir = Join-Path $env:S3_CAFEBERT_RESULTS_DIR "representation_cache"
New-Item -ItemType Directory -Force -Path $NewCacheDir | Out-Null
$ReferenceCacheDir = Join-Path $RepoRoot "benchmark\cafebert_full\reference\representation_cache"
if (Test-Path $ReferenceCacheDir) {
    Invoke-Step -Name "Seed representation_cache from reference/ (skip re-encoding documents)" -Body {
        Copy-Item -Path (Join-Path $ReferenceCacheDir "*.npy") -Destination $NewCacheDir -Force -ErrorAction Stop
        # representation_for_corpus() only treats a cached .npy as a hit if a
        # matching <same-stem>.json sidecar ALSO exists (it stores
        # cold_representation_seconds there). reference/representation_cache/
        # never committed those sidecars (only .npy + _lexical.npz) -- a plain
        # copy of *.json finds nothing, silently leaving the .npy alone and
        # forcing a full re-encode. Synthesize one sidecar per .npy instead;
        # the timing value inside is informational only (not used by
        # run_cafebert_refit_optimized.py), so a placeholder is fine.
        # Windows PowerShell 5.1's `Set-Content -Encoding utf8` writes a UTF-8
        # BOM, which Python's json.loads() does not skip -- write via .NET
        # directly with a BOM-less UTF8Encoding instead, or json.loads() on
        # the Python side throws "Unexpected UTF-8 BOM" exactly like it just did.
        $Utf8NoBom = New-Object System.Text.UTF8Encoding($false)
        Get-ChildItem -Path $NewCacheDir -Filter "*.npy" | ForEach-Object {
            $sidecar = Join-Path $NewCacheDir ($_.BaseName + ".json")
            if (-not (Test-Path $sidecar)) {
                $json = @{ cold_representation_seconds = 0.0; note = "synthesized sidecar to force cache hit on the copied reference .npy" } | ConvertTo-Json
                [System.IO.File]::WriteAllText($sidecar, $json, $Utf8NoBom)
            }
        }
    }
}

# --- 7. The actual optimized re-run: full grid, all 4 corpora, all 4 seeds, all 5 k. ---
Invoke-Step -Name "run_cafebert_refit_optimized (full grid)" -Fatal $true -Body {
    Invoke-Native $VenvPython -m benchmark.cafebert_full.run_cafebert_refit_optimized
}

# --- 8. Comparison report for Claude to read back when updating report/paper.tex ---
Invoke-Step -Name "compare_refit_speedup (write REFIT_SPEEDUP_COMPARISON.md)" -Body {
    Invoke-Native $VenvPython -m benchmark.cafebert_full.compare_refit_speedup
}

Write-Log "=== ALL STEPS DONE. Results: benchmark\cafebert_full\results_refit\ ==="
Write-Log "Read benchmark\cafebert_full\results_refit\REFIT_SPEEDUP_COMPARISON.md first."
Write-Log "Full log saved at: $LogFile"
