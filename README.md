# Almond-Morphometric-Prediction

Supplementary materials, analysis code, model outputs, and figures associated with the study:

**Machine Learning-Based Prediction of Al-Shami Almond Kernel Traits from External Fruit Morphometric Features**

## Supplementary Materials

### Supplementary File S1 – Experiment 1

Multiple linear regression analysis for predicting kernel morphometric ratios (Ker_L/W, Ker_W/T, and Ker_L/T) from external fruit morphometric ratios (L/W, W/T, and L/T).

### Supplementary File S2 – Experiment 2

Multiple linear regression analysis for predicting average fruit mass (FruitMass_Avg) from external fruit morphometric ratios (L/W, W/T, and L/T).

### Supplementary File S3 – Experiment 3

Multiple linear regression analysis for predicting hull thickness measurements from external fruit morphometric ratios (L/W, W/T, and L/T).

Each supplementary folder contains the corresponding analysis code, model output, performance metrics, and supplementary figures.

## Support Vector Regression (SVR) Analysis

The SVR folder contains the dataset, Python analysis code, model outputs, and figures associated with the Support Vector Regression analysis for predicting kernel morphometric ratios (Ker_L/W, Ker_W/T, and Ker_L/T) from external fruit morphometric ratios (L/W, W/T, and L/T).

Linear, polynomial, and radial basis function (RBF) kernels were evaluated using 5-fold cross-validation and hyperparameter optimization. The optimized RBF-SVR model was subsequently evaluated using an independent test set.

The SVR folder includes:

- Input dataset
- Python analysis code
- Hyperparameter optimization and model performance results
- Actual versus predicted plots
- Residual plots
- Permutation-based feature importance results and plots
- Fruit-to-kernel morphometric relationship plots

## Software

Analyses were performed in Python 3 using the scikit-learn library.

## Citation

If using these materials, please cite the associated article.

Citation information will be updated following publication.
