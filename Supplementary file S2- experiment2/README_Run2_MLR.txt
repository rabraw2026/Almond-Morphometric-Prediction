RUN 2 — MULTIPLE LINEAR REGRESSION: FRUIT MASS

Purpose
-------
Predict average fruit mass (FruitMass_Avg) from the external fruit morphometric
ratios L/W, W/T, and L/T using multiple linear regression.

Data split
----------
100 observations; 80% training and 20% independent test set (random_state = 42).

Performance
-----------
Test R²  = -0.1374262630
Test MSE = 2.0893098669

Recommended supplementary submission files
-------------------------------------------
Run2_Description.txt
Run2_MLR_Code.txt
Run2_Performance_Metrics.txt
Run2_Complete_Output.txt
Supplementary_Figure_S2_MLR_Observed_vs_Predicted_Fruit_Mass.png
Supplementary_Figure_S3_MLR_Residuals_Fruit_Mass.png

Optional archive files
----------------------
Run2_MLR_Coefficients_Fruit_Mass.png
Run2_Complete_Output.html

Note
----
The coefficient plot is not labelled as 'feature importance' in the submission
set because these are raw multiple-linear-regression coefficients, not
permutation-based feature-importance scores. The HTML output is retained only
for archival completeness because the TXT output already provides the relevant
analysis output.
