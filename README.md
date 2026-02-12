# Evaluating the Confidence-Interval Performance of the Double LASSO Estimator in High-Dimensional Linear Models

This project studies the finite-sample behavior of the Double LASSO estimator for inference in high-dimensional linear regression models. Using controlled Monte Carlo simulations, we examine how confidence-interval coverage and length behave under different data-generating processes (Gaussian, heavy-tailed, and approximate-sparse) and varying levels of dimensionality and covariate correlation. The simulations assess how regularization choices—plug-in versus cross-validated penalties affect the stability and reliability of post-selection inference relative to ordinary least squares in settings where covariates are numerous relative to sample size. 

## Project structure
- `scenarios.py`: scenario grid controlling sample size, dimensionality, sparsity, correlation, and noise.
- `dgps/`: data-generating processes (`static`, `static_easier`, `heavy_tail`).
- `estimators/`: Double LASSO (plug-in and CV penalties) and OLS baseline.
- `main.py` / `runner.py`: simulation entry point and execution helpers.
- `plots.py` / `plots_compare.py`: plotting and comparison utilities.
- `results*/`: simulation outputs (CSV summaries) for different designs/penalty settings.
- `article.qmd` + `article_files/`: Quarto manuscript and rendered assets.
- `double-lasso-presentation.qmd` + `double-lasso-presentation_files/`: Quarto slides and assets.
- `pdf-title.tex` / `pdf-no-title.tex`: LaTeX title-page variants.
- `slides.css`: custom slide styling.

## Installation
Install the required Python dependencies:
```bash
python3 -m pip install -r requirements.txt
```

## Manuscript rendering
The manuscript is written in Quarto (`article.qmd`). To render it, install Quarto and a LaTeX distribution (e.g., TeX Live or MacTeX). These are external tools and are not included in `requirements.txt`.

## Reproducibility
Seeds are fixed at the scenario level and can be overridden via CLI flags. Given the same seed, code version, and environment, results are deterministic.

## Running the simulations (all scenarios + plots)

From the project root:
```bash
python3 main.py
```
This runs all predefined scenario grids for each DGP (plug-in and CV penalties), runs the OLS baseline, and generates plots.

### Build the PDF
From the project root:
```bash
quarto render article.qmd --to pdf
```
This produces `article.pdf` in the project root.

## Authors
Shokhrukhkhon Nishonkulov  
Olimjon Umurzokov  
Damir Abdulazizov  
M.Sc. Economics, University of Bonn  
Research Module in Econometrics and Statistics (2025)  
Professor: Vladislav Morozov
