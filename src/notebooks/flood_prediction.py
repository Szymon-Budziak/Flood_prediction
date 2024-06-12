# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: .venv
#     language: python
#     name: python3
# ---

# ## 1. Packages import and data loading

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle as pkl

df_train = pd.read_csv('../data/train.csv')

print(f'Shape of train data: {df_train.shape}')

# ## 2. Data exploration

df_train.head()

# Description of each column in the dataset:
# 1. **id**: Unique identifier for each record in the dataset.
# 2. **MonsoonIntensity**: Index representing monsoon intensity. Higher values indicate greater monsoon rainfall intensity. (strong temporary wind)
# 3. **TopographyDrainage**: Index representing the effectiveness of topographic drainage. Higher values indicate better drainage.
# 4. **RiverManagement**: Index representing the effectiveness of river management. Higher values indicate more effective management.
# 5. **Deforestation**: Index representing the level of deforestation. Higher values indicate greater deforestation.
# 6. **Urbanization**: Index representing the level of urbanization. Higher values indicate greater urbanization.
# 7. **ClimateChange**: Index representing the impact of climate change. Higher values indicate greater impact.
# 8. **DamsQuality**: Index representing the quality of dams. Higher values indicate better quality of dams. (tama)
# 9. **Siltation**: Index representing the level of siltation. Higher values indicate greater siltation. (zamulenie)
# 10. **AgriculturalPractices**: Index representing agricultural practices. Higher values indicate more intensive or less sustainable practices.
# 11. **Encroachments**: Index representing the level of illegal or unauthorized land occupation. Higher values indicate greater levels of occupation.
# 12. **IneffectiveDisasterPreparedness**: Index representing the ineffectiveness of disaster preparedness. Higher values indicate less effective preparedness.
# 13. **DrainageSystems**: Index representing the effectiveness of drainage systems. Higher values indicate more effective drainage systems.
# 14. **CoastalVulnerability**: Index representing coastal vulnerability. Higher values indicate greater vulnerability of coastal areas.
# 15. **Landslides**: Index representing the probability of landslides. Higher values indicate greater likelihood of landslides.
# 16. **Watersheds**: Index representing watershed management. Higher values indicate more effective management.
# 17. **DeterioratingInfrastructure**: Index representing the deterioration of infrastructure. Higher values indicate greater deterioration. (pogorszenie)
# 18. **PopulationScore**: Index representing population density. Higher values indicate greater population density.
# 19. **WetlandLoss**: Index representing the loss of wetlands. Higher values indicate greater loss of wetlands. (tereny podmokle)
# 20. **InadequatePlanning**: Index representing inadequate planning. Higher values indicate less adequate planning.
# 21. **PoliticalFactors**: Index representing the impact of political factors. Higher values indicate greater impact of political factors.
# 22. **FloodProbability**: Flood probability, expressed as a continuous value between 0 and 1, where higher values indicate greater flood probability.
#
# These columns represent variables that can influence the flood probability of a region. We would like to predict **FloodProbability** based on other features.

df_train.info()

# check for nan values in training set in each column
df_train.isnull().sum(axis=0)

# We can observe that there are no NaN values in this dataset. Also there are no categorical variables, only numerical. This allows us to skip the step of encoding categorical variables into numerical ones using one-hot encoding.

# ## 3. Statistical analysis of the varaibles

# #### 3.1 Describe method on training set

df_train.describe(include='all')

# #### 3.2 Observations
#
# 1. **Count**: All variables have the same number of observations, 1,117,957, indicating no missing values in the dataset.
# 2. **Mean**: The mean of variables like *MonsoonIntensity*, *TopographyDrainage*, *RiverManagement*, and others is around 5, suggesting fairly uniform distribution of values.
#  - The mean of *FloodProbability* is about 0.504, indicating that floods are relatively common.
# 3. **Standard Deviation (std)**:
#  - The standard deviation for many variables is around 2, indicating some variability in the data.
#  - The standard deviation for *FloodProbability* is 0.051, indicating lower variability for this variable.
# 4. **Minimum and Maximum (min and max)**:
#  - Most variables have minimum values of 0 and maximum values around 16-18.
#  - *FloodProbability* has a minimum of 0.285 and a maximum of 0.725, indicating that flood probability does not reach extremes.
# 5. **Quartiles (25%, 50%, 75%)**:
#  - The quartiles of the variables show that values are uniformly distributed, with half of the data between 3 and 6 for many variables.
#  - *FloodProbability* has a median (50%) of about 0.505, which is very close to the mean
#
# #### 3.3 Key deduction
#  - Variables have values centered around 5.
#  - The standard deviation is relatively low compared to the scale of values.
#  - Variables like FloodProbability are already normalized between 0 and 1.
#  - There are no missing values in the dataset.
#
# #### 3.4 Specific activities
#  - **Data Preprocessing**: Since the data does not have missing values and appears to be normalized on a common scale, we can proceed directly to data splitting and standardization if necessary.
#  - **Variance Evaluation**: All variables have similar variance, so we might consider dimensionality reduction techniques like PCA only if necessary to improve model performance.
#
# #### 3.5 Data distribution and variance
#  - Variables in the dataset are already centered around intermediate values (about 5) and have a distribution that does not show clear skewness requiring logarithmic transformation.
#  - Data Variance: The standard deviations of the variables are relatively low compared to their mean values, suggesting that the data does not have an extremely wide range of values that might benefit from logarithmic transformation.

# ## 4. Distribution of numerical variables

numerical_cols = df_train.select_dtypes(include=['float64', 'int64']).columns.drop("id")
print(f'Numerical columns (length: {len(numerical_cols)}):\n', numerical_cols)

# +
plt.figure(figsize=(20, 20))

for i, col in enumerate(numerical_cols):
    plt.subplot(6, 4, i+1)
    sns.histplot(df_train[col], bins=15, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Histograms of numerical features in train data')
plt.savefig('../images/histograms_numerical_features.png')
plt.show()
# -

# Following the analysis of the numerical variables' distributions in the training set, we can draw several conclusions:
# #### MonsoonIntensity, TopographyDrainage, RiverManagement, etc. (descriptive featues):
#  - Most variables show unimodal or bimodal distributions with various dispersions.
#  - Variables like MonsoonIntensity, TopographyDrainage, RiverManagement, and others show fairly uniform distribution with peaks at certain values, with some outliers.
# #### FloodProbability (target feature):
#  - The target variable **FloodProbability** seems to have a fairly normal distribution with a slight skew.
#  - This could suggest that the dataset is fairly balanced with respect to flood probability.

# ## 5. Feature scaling

from sklearn.preprocessing import StandardScaler

# drop target feature - FloodProbability for feature scaling
numerical_cols = numerical_cols.drop("FloodProbability")

# +
# scale numerical features
scaler = StandardScaler()

df_train[numerical_cols] = scaler.fit_transform(df_train[numerical_cols])
# -

# check if scaling was successful
df_train.describe(include='all')

# +
# plot the new distributions of the numerical features after scaling
plt.figure(figsize=(20, 20))

for i, col in enumerate(numerical_cols):
    plt.subplot(6, 4, i+1)
    sns.histplot(df_train[col], bins=15, kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel(col)
    plt.ylabel('Frequency')

plt.tight_layout()
plt.suptitle('Transformed histograms of numerical features in train data')
plt.savefig('../images/histograms_numerical_features_transformed.png')
plt.show()
# -

# ## 6. Model creation

# Tested models:
#  - Linear Regression
#  - Elastic Net Regression
#  - Support Vector Regressor
#  - KNeighbors Regressor
#  - Decision Tree Regressor

# #### 6.1 Train/Test split

# +
from sklearn.model_selection import train_test_split

X = df_train.drop(columns=["id", "FloodProbability"], axis=1)
y = df_train["FloodProbability"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")
# -

# #### 6.2 K-fold evaluation function
#
#  - **KFold cross-validation** to evaluate the performance of models on different splits of the training data and choose the best model.
#
#  - Calculation and Visualization of Results: We calculate the average MSE and R2 across all folds of the cross-validation and visualize them.
#
#  - **GridSearchCV**: Used to search for the best hyperparameters within a defined grid.

# +
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import KFold

all_metrics =[]

def evaluate_with_kfold(
    model, model_name: str, n_splits: int, X_: pd.DataFrame, y_: pd.DataFrame
) -> tuple[float, float]:
    print(f'Cross-validation with {model_name} started...')
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    cv_mse = []
    cv_r2 = []

    fold_n = 1
    for train_idx, val_idx in kf.split(X_):
        X_tr, X_val = X_.iloc[train_idx], X_.iloc[val_idx]
        y_tr, y_val = y_.iloc[train_idx], y_.iloc[val_idx]

        model.fit(X_tr, y_tr)
        y_pred_val = model.predict(X_val)

        val_mse = mean_squared_error(y_val, y_pred_val)
        val_r2 = r2_score(y_val, y_pred_val)

        print(f'Fold {fold_n}:\n\tValidation MSE: {val_mse}\n\tValidation R^2: {val_r2}')

        cv_mse.append(val_mse)
        cv_r2.append(val_r2)

        fold_n += 1

    avg_mse = np.mean(cv_mse)
    avg_r2 = np.mean(cv_r2)

    print(f'Num of splits: {n_splits}\n\tAverage Validation MSE: {avg_mse}\n\tAverage Validation R^2: {avg_r2}')

    return avg_mse, avg_r2


# -

# #### 6.3 Linear Regression

# +
from sklearn.linear_model import LinearRegression

model_name, model = 'Linear Regression', LinearRegression()
mse_lin, r2_lin = evaluate_with_kfold(model, model_name, 10, X_train, y_train)
all_metrics.append((model_name, mse_lin, r2_lin))
# -

# #### 6.4 Elastice Net Regression

# +
from sklearn.linear_model import ElasticNet

model_name, model = 'Elastic Net', ElasticNet()
mse_en, r2_en = evaluate_with_kfold(model, model_name, 10, X_train, y_train)
all_metrics.append((model_name, mse_en, r2_en))
# -

# #### 6.5 Support Vector Regressor

# +
from sklearn.svm import SVR

model_name, model = 'Support Vector Regressor', SVR()
mse_svr, r2_svr = evaluate_with_kfold(model, model_name, 10, X_train, y_train)
all_metrics.append((model_name, mse_svr, r2_svr))
# -

# #### 6.6 KNeighbors Regressor

# +
from sklearn.neighbors import KNeighborsRegressor

model_name, model = 'K-Nearest Neighbors', KNeighborsRegressor()
mse_knn, r2_knn = evaluate_with_kfold(model, model_name, 10, X_train, y_train)
all_metrics.append((model_name, mse_knn, r2_knn))
# -

# #### 6.7 Decision Tree Regressor

# +
from sklearn.tree import DecisionTreeRegressor

model_name, model = 'Decision Tree', DecisionTreeRegressor()
mse_dt, r2_dt = evaluate_with_kfold(model, model_name, 10, X_train, y_train)
all_metrics.append((model_name, mse_dt, r2_dt))
# -

# save metrics to pickle file
with open('../data/all_metrics.pkl', 'wb') as f:
    pkl.dump(all_metrics, f)

# load metrics from pickle file
with open('../data/all_metrics.pkl', 'rb') as f:
    all_metrics = pkl.load(f)

# #### 6.8 Visualize models MSE and R2

df_metrics = pd.DataFrame(all_metrics, columns=["Model", "MSE", "R^2"]).set_index("Model")
df_metrics

# metrics sorted by lowest MSE
df_metrics.sort_values(by="MSE", ascending=True)

# metrics sorted by highest R^2
df_metrics.sort_values(by="R^2", ascending=False)

# plot the MSE of the models
plt.figure(figsize=(10, 6))
plt.bar(df_metrics.index, df_metrics["MSE"], color='blue')
plt.ylabel("MSE")
plt.title("Mean Squared Error of the models")
plt.xticks(rotation=45)
plt.savefig('../images/mse_models.png')
plt.show()

# plot the R^2 score of the models
plt.figure(figsize=(10, 6))
plt.bar(df_metrics.index, df_metrics["R^2"], color='red')
plt.ylabel("R^2")
plt.title("R^2 score of the models")
plt.xticks(rotation=45)
plt.savefig('../images/r2_models.png')
plt.show()

# We can observe that the best model is **Linear Regression** with MSE value of `0.000404` and R^2 `0.844957`.

# ## 7. Hyperparameter tunning

# +
from sklearn.model_selection import GridSearchCV

param_grid = {
    'fit_intercept': [True, False],
    'copy_X': [True, False],
    'positive': [True, False]
}

grid_search = GridSearchCV(LinearRegression(), param_grid=param_grid, cv=10, scoring='neg_mean_squared_error', n_jobs=-1)
grid_search.fit(X_train, y_train)

# best hyperparameters
best_params = grid_search.best_params_
print(f'Best hyperparameters: {best_params}')

# best model
best_model = grid_search.best_estimator_
print(f'Best model: {best_model}')
# -

mse_best, r2_best = evaluate_with_kfold(best_model, 'Best model', 10, X_test, y_test)

df_metrics.loc['Best model'] = [mse_best, r2_best]

# +
# plot the MSE of the models
best_mse = df_metrics["MSE"].sort_values()

plt.figure(figsize=(10, 6))
plt.bar(best_mse.index, best_mse, color="blue")
plt.ylabel("MSE")
plt.title("Mean Squared Error of the models")
plt.xticks(rotation=45)
plt.savefig("../images/mse_all_models.png")
plt.show()

# +
# plot the R^2 score of the models
best_r2 = df_metrics["R^2"].sort_values(ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(best_r2.index, best_r2, color="red")
plt.ylabel("R^2")
plt.title("R^2 score of the models")
plt.xticks(rotation=45)
plt.savefig("../images/r2_all_models.png")
plt.show()
