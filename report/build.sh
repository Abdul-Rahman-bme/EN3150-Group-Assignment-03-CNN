#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
pdflatex -interaction=nonstopmode -halt-on-error -jobname=54_A03_EN3150 main.tex
bibtex 54_A03_EN3150
pdflatex -interaction=nonstopmode -halt-on-error -jobname=54_A03_EN3150 main.tex
pdflatex -interaction=nonstopmode -halt-on-error -jobname=54_A03_EN3150 main.tex
