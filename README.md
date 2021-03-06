This repository stores the code and experimental results for the comparison of RENT feature selection with BIC hyperparameter search [1] and state-of-the-art feature selectors, including the stability selection method [2] implemented in the stabs package in R [3]. It contains the code to reproduce experimental results for Section III E in [4]. Both methods perform ensemble feature selection based on generalized linear models with regularization. The experimental comparison is done on two datasets, denoted as c0 (synthetic dataset generated using scikit-learn method make_classification) and c3 (Breast Cancer Wisconsin dataset [5]) following the naming convention in [4]. The deployed metrics evaluate predictive performance (F1, Matthews correlation coefficient).

Our work contains 4 Jupyter notebooks: two for each dataset (c0 and c3, respectively), where the file with suffix "python" calls RENT, RFE, reliefF and Random Forest, and the file with suffix "R" calls stability selection, Laplace score, Fisher score and mRMR. While RENT performs hyperparameter selection by minimizing the Bayesian information criterion (BIC), see BIC_hyperparameter_search.py, stability selection applies hyperparameter selection via grid search on validation data, see stabsel_gridsearch.R. Predictive performances are calculated by training an unregularized GLM after feature selection.

[1] Jenul et al., (2021). RENT: A Python Package for Repeated Elastic Net Feature Selection. Journal of Open Source Software, 6(63), 3323, https://doi.org/10.21105/joss.03323

[2] Meinshausen, N. and Bühlmann, P. (2010), Stability selection. Journal of the Royal Statistical Society: Series B (Statistical Methodology), 72: 417-473, https://doi.org/10.1111/j.1467-9868.2010.00740.x

[3] Hofner, B. and Hothorn, T. (2021). stabs: Stability Selection with Error Control. R package version 0.6-4, https://CRAN.R-project.org/package=stabs

[4] Jenul et al., (2021). RENT -- Repeated Elastic Net Technique for Feature Selection. Manuscript under review.

[5] Wolberg, W., Street, W. and Mangasarian, O. (1995). Breast Cancer Wisconsin (Diagnostic). UCI Machine Learning Repository.
