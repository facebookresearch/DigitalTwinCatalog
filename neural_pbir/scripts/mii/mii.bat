@REM Copyright (c) Meta Platforms, Inc. and affiliates.
@REM All rights reserved.
@REM
@REM This source code is licensed under the license found in the
@REM LICENSE file in the root directory of this source tree.

@echo off
rem ===============================================================
rem  run_mii_pipeline.bat  – Windows replacement for the Bash script
rem ===============================================================
rem Assumes:
rem   * You’re running a 64-bit Anaconda / Python prompt
rem   * The current working directory is the repo root
rem   * All relative paths below exist exactly as in the Unix repo
rem ---------------------------------------------------------------

:: ----- variables ------------------------------------------------
setlocal EnableDelayedExpansion

set "DATAROOT=data\Synthetic4Relight"
set "CKPTROOT=results\mii"

:: Surface reconstruction template
set "MII_CMD=python neural_surface_recon\run_template.py --template neural_surface_recon\configs\template_mii.py --savemem"

:: PBIR template
set "PBIR_CMD=python pbir\run.py pbir\configs\template"

:: ----- list of scenes ------------------------------------------
set "SCENES=air_baloons chair hotdog jugs"

echo === Stage 1: neural_surface_recon ========================================
for %%S in (%SCENES%) do (
    echo Running MII for %%S …
    call %MII_CMD% "%DATAROOT%\%%S"
)

echo === Stage 2: neural_distillation ========================================
for %%S in (%SCENES%) do (
    echo Distilling %%S …
    call python neural_distillation\run.py "%CKPTROOT%\%%S"
)

echo === Stage 3: PBIR evaluation ============================================
for %%S in (%SCENES%) do (Y
    echo PBIR on %%S …
    call %PBIR_CMD% "%CKPTROOT%\%%S"
)

echo.
echo -------- Pipeline finished --------
endlocal
