# Feature Importance Analysis Report
## Overall Summary by Window Size

### Window Size ±0
Top features by importance across models:
- sasa: Found in 6 models: random_forest (0.116), xgboost (0.091), lightgbm (393.000), gradient_boosting (0.156), logistic (0.054), shap (0.228)
- chi1: Found in 5 models: random_forest (0.115), xgboost (0.085), lightgbm (334.000), gradient_boosting (0.183), shap (0.237)
- omega: Found in 2 models: random_forest (0.111), gradient_boosting (0.123)
- ss: Found in 2 models: xgboost (0.094), logistic (0.140)

### Window Size ±1
Top features by importance across models:
- sasa: Found in 6 models: random_forest (0.046), xgboost (0.028), lightgbm (134.333), gradient_boosting (0.087), logistic (0.183), shap (0.184)
- chi1: Found in 2 models: random_forest (0.039), lightgbm (129.667)
- chi4: Found in 3 models: xgboost (0.052), gradient_boosting (0.074), shap (0.144)
- ss: Found in 2 models: xgboost (0.032), logistic (0.080)
- chi2: Found in 3 models: lightgbm (132.333), gradient_boosting (0.037), shap (0.131)

### Window Size ±2
Top features by importance across models:
- sasa: Found in 6 models: random_forest (0.029), xgboost (0.018), lightgbm (79.400), gradient_boosting (0.062), logistic (0.168), shap (0.142)
- chi1: Found in 2 models: random_forest (0.023), lightgbm (84.200)
- chi4: Found in 3 models: xgboost (0.029), gradient_boosting (0.043), shap (0.095)
- ss: Found in 2 models: xgboost (0.018), logistic (0.064)
- chi2: Found in 3 models: lightgbm (83.400), gradient_boosting (0.021), shap (0.113)

### Window Size ±4
Top features by importance across models:
- sasa: Found in 6 models: random_forest (0.016), xgboost (0.012), lightgbm (47.111), gradient_boosting (0.033), logistic (0.112), shap (0.093)
- chi1: Found in 2 models: random_forest (0.013), lightgbm (44.000)
- chi4: Found in 4 models: xgboost (0.018), gradient_boosting (0.024), logistic (0.053), shap (0.069)
- chi2: Found in 3 models: lightgbm (50.000), gradient_boosting (0.012), shap (0.071)

### Window Size ±8
Top features by importance across models:
- sasa: Found in 6 models: random_forest (0.009), xgboost (0.006), lightgbm (26.294), gradient_boosting (0.017), logistic (0.077), shap (0.063)
- tau: Found in 2 models: random_forest (0.007), logistic (0.053)
- chi4: Found in 3 models: xgboost (0.009), gradient_boosting (0.015), shap (0.060)
- chi2: Found in 3 models: lightgbm (22.706), gradient_boosting (0.005), shap (0.048)

### Window Size ±16
Top features by importance across models:
- sasa: Found in 6 models: random_forest (0.004), xgboost (0.004), lightgbm (14.788), gradient_boosting (0.008), logistic (0.055), shap (0.041)
- chi4: Found in 4 models: xgboost (0.004), gradient_boosting (0.007), logistic (0.057), shap (0.032)
- chi2: Found in 2 models: lightgbm (11.333), shap (0.030)
- chi1: Found in 2 models: lightgbm (11.030), gradient_boosting (0.003)

## Model-Specific Analysis

### RANDOM_FOREST Analysis

phi:
- Optimal window size: ±0
- Maximum importance: 0.105
- Trend: Importance decreases with window size

psi:
- Optimal window size: ±0
- Maximum importance: 0.106
- Trend: Importance decreases with window size

omega:
- Optimal window size: ±0
- Maximum importance: 0.111
- Trend: Importance decreases with window size

tau:
- Optimal window size: ±0
- Maximum importance: 0.107
- Trend: Importance decreases with window size

chi1:
- Optimal window size: ±0
- Maximum importance: 0.115
- Trend: Importance decreases with window size

chi2:
- Optimal window size: ±0
- Maximum importance: 0.108
- Trend: Importance decreases with window size

chi3:
- Optimal window size: ±0
- Maximum importance: 0.105
- Trend: Importance decreases with window size

chi4:
- Optimal window size: ±0
- Maximum importance: 0.107
- Trend: Importance decreases with window size

sasa:
- Optimal window size: ±0
- Maximum importance: 0.116
- Trend: Importance decreases with window size

ss:
- Optimal window size: ±0
- Maximum importance: 0.006
- Trend: Importance decreases with window size

### XGBOOST Analysis

phi:
- Optimal window size: ±0
- Maximum importance: 0.076
- Trend: Importance decreases with window size

psi:
- Optimal window size: ±0
- Maximum importance: 0.074
- Trend: Importance decreases with window size

omega:
- Optimal window size: ±0
- Maximum importance: 0.078
- Trend: Importance decreases with window size

tau:
- Optimal window size: ±0
- Maximum importance: 0.082
- Trend: Importance decreases with window size

chi1:
- Optimal window size: ±0
- Maximum importance: 0.085
- Trend: Importance decreases with window size

chi2:
- Optimal window size: ±0
- Maximum importance: 0.079
- Trend: Importance decreases with window size

chi3:
- Optimal window size: ±0
- Maximum importance: 0.080
- Trend: Importance decreases with window size

chi4:
- Optimal window size: ±0
- Maximum importance: 0.075
- Trend: Importance decreases with window size

sasa:
- Optimal window size: ±0
- Maximum importance: 0.091
- Trend: Importance decreases with window size

ss:
- Optimal window size: ±0
- Maximum importance: 0.094
- Trend: Importance decreases with window size

### LIGHTGBM Analysis

phi:
- Optimal window size: ±0
- Maximum importance: 293.000
- Trend: Importance decreases with window size

psi:
- Optimal window size: ±0
- Maximum importance: 311.000
- Trend: Importance decreases with window size

omega:
- Optimal window size: ±0
- Maximum importance: 323.000
- Trend: Importance decreases with window size

tau:
- Optimal window size: ±0
- Maximum importance: 281.000
- Trend: Importance decreases with window size

chi1:
- Optimal window size: ±0
- Maximum importance: 334.000
- Trend: Importance decreases with window size

chi2:
- Optimal window size: ±0
- Maximum importance: 360.000
- Trend: Importance decreases with window size

chi3:
- Optimal window size: ±0
- Maximum importance: 332.000
- Trend: Importance decreases with window size

chi4:
- Optimal window size: ±0
- Maximum importance: 333.000
- Trend: Importance decreases with window size

sasa:
- Optimal window size: ±0
- Maximum importance: 393.000
- Trend: Importance decreases with window size

ss:
- Optimal window size: ±0
- Maximum importance: 13.333
- Trend: Importance decreases with window size

### GRADIENT_BOOSTING Analysis

phi:
- Optimal window size: ±0
- Maximum importance: 0.045
- Trend: Importance decreases with window size

psi:
- Optimal window size: ±0
- Maximum importance: 0.099
- Trend: Importance decreases with window size

omega:
- Optimal window size: ±0
- Maximum importance: 0.123
- Trend: Importance decreases with window size

tau:
- Optimal window size: ±0
- Maximum importance: 0.094
- Trend: Importance decreases with window size

chi1:
- Optimal window size: ±0
- Maximum importance: 0.183
- Trend: Importance decreases with window size

chi2:
- Optimal window size: ±0
- Maximum importance: 0.105
- Trend: Importance decreases with window size

chi3:
- Optimal window size: ±0
- Maximum importance: 0.084
- Trend: Importance decreases with window size

chi4:
- Optimal window size: ±0
- Maximum importance: 0.100
- Trend: Importance decreases with window size

sasa:
- Optimal window size: ±0
- Maximum importance: 0.156
- Trend: Importance decreases with window size

ss:
- Optimal window size: ±0
- Maximum importance: 0.004
- Trend: Importance decreases with window size

### LOGISTIC Analysis

phi:
- Optimal window size: ±1
- Maximum importance: 0.038
- Trend: Importance decreases with window size

psi:
- Optimal window size: ±4
- Maximum importance: 0.041
- Trend: Importance increases with window size

omega:
- Optimal window size: ±0
- Maximum importance: 0.041
- Trend: Importance decreases with window size

tau:
- Optimal window size: ±0
- Maximum importance: 0.114
- Trend: Importance decreases with window size

chi1:
- Optimal window size: ±0
- Maximum importance: 0.041
- Trend: Importance decreases with window size

chi2:
- Optimal window size: ±8
- Maximum importance: 0.027
- Trend: Importance increases with window size

chi3:
- Optimal window size: ±0
- Maximum importance: 0.043
- Trend: Importance decreases with window size

chi4:
- Optimal window size: ±16
- Maximum importance: 0.057
- Trend: Importance increases with window size

sasa:
- Optimal window size: ±1
- Maximum importance: 0.183
- Trend: Importance increases with window size

ss:
- Optimal window size: ±0
- Maximum importance: 0.140
- Trend: Importance decreases with window size

### SHAP Analysis

phi:
- Optimal window size: ±0
- Maximum importance: 0.190
- Trend: Importance decreases with window size

psi:
- Optimal window size: ±0
- Maximum importance: 0.205
- Trend: Importance decreases with window size

omega:
- Optimal window size: ±0
- Maximum importance: 0.201
- Trend: Importance decreases with window size

tau:
- Optimal window size: ±0
- Maximum importance: 0.178
- Trend: Importance decreases with window size

chi1:
- Optimal window size: ±0
- Maximum importance: 0.237
- Trend: Importance decreases with window size

chi2:
- Optimal window size: ±0
- Maximum importance: 0.189
- Trend: Importance decreases with window size

chi3:
- Optimal window size: ±0
- Maximum importance: 0.173
- Trend: Importance decreases with window size

chi4:
- Optimal window size: ±0
- Maximum importance: 0.185
- Trend: Importance decreases with window size

sasa:
- Optimal window size: ±0
- Maximum importance: 0.228
- Trend: Importance decreases with window size

ss:
- Optimal window size: ±0
- Maximum importance: 0.014
- Trend: Importance decreases with window size

## Feature-Specific Insights

### phi
- Average importance across all models and windows: 14.703 (±51.312)
- Best performance: 293.000 with lightgbm at window size ±0

### psi
- Average importance across all models and windows: 14.707 (±53.892)
- Best performance: 311.000 with lightgbm at window size ±0

### omega
- Average importance across all models and windows: 16.559 (±57.128)
- Best performance: 323.000 with lightgbm at window size ±0

### tau
- Average importance across all models and windows: 13.513 (±48.746)
- Best performance: 281.000 with lightgbm at window size ±0

### chi1
- Average importance across all models and windows: 17.408 (±59.421)
- Best performance: 334.000 with lightgbm at window size ±0

### chi2
- Average importance across all models and windows: 18.361 (±63.481)
- Best performance: 360.000 with lightgbm at window size ±0

### chi3
- Average importance across all models and windows: 14.038 (±55.991)
- Best performance: 332.000 with lightgbm at window size ±0

### chi4
- Average importance across all models and windows: 14.567 (±56.432)
- Best performance: 333.000 with lightgbm at window size ±0

### sasa
- Average importance across all models and windows: 19.363 (±68.401)
- Best performance: 393.000 with lightgbm at window size ±0

### ss
- Average importance across all models and windows: 0.544 (±2.216)
- Best performance: 13.333 with lightgbm at window size ±0
