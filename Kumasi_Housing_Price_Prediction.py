# Kumasi Ghana Housing Price Prediction using Machine Learning
# Generated from the notebook version and following the referenced house-pricing-regression workflow.

# # Kumasi Ghana Housing Price Prediction using Machine Learning
# 
# This notebook follows the structure of the referenced **house-pricing-regression** notebook:
# **Read the data → Inspect the data → EDA → Dummy Variables → Train/Test Split → Scaling → RFE → OLS/VIF → Residual Analysis → Test Prediction → Evaluation**.
# 
# The data used here is the supplied **Kumasi Ghana Housing dataset**. The price target is represented in **Ghana cedis (GH₵)**.

# > **Project note:** This dataset is a synthetic/modified learning dataset for a Kumasi housing-price machine-learning project. Model results describe this supplied dataset and should not be interpreted as live market valuation.

# ## Read the data

import warnings
warnings.filterwarnings("ignore")

# Auto-install missing packages before importing them.
import importlib
import subprocess
import sys

REQUIRED_PACKAGES = [
    "numpy",
    "pandas",
    "matplotlib",
    "seaborn",
    "scikit-learn",
    "statsmodels",
    "scipy",
    "joblib",
]

for package in REQUIRED_PACKAGES:
    try:
        importlib.import_module(package)
    except ModuleNotFoundError:
        print(f"Installing missing package: {package}")
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "--quiet", package]
        )

from pathlib import Path
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Find the supplied dataset whether this notebook is run beside the CSV
# or from the current project working directory.
possible_paths = [
    Path("Kumasi_Ghana_Housing.csv"),
    Path("Kumasi_Ghana_Housing(1).csv"),
    Path("/mnt/data/Kumasi_Ghana_Housing.csv"),
    Path("/mnt/data/Kumasi_Ghana_Housing(1).csv")
]

DATA_PATH = next((p for p in possible_paths if p.exists()), None)

if DATA_PATH is None:
    raise FileNotFoundError(
        "Kumasi Ghana Housing CSV not found. Put the CSV in the same folder as this notebook."
    )

df = pd.read_csv(DATA_PATH)
print(f"Loaded: {DATA_PATH}")

# ### Display some rows

df.head()

# ### View the number of rows and columns in the data

df.shape

# ### View summary statistics

df.describe()

# ### View data information

df.info()

# ### Look if there are any duplicate rows in the dataset

df.loc[df.duplicated()]

# No duplicate rows are expected in this supplied dataset.

# ### Get the name of the columns

df.columns.tolist()

# ### Check for null values

df.isna().sum()

# The supplied dataset contains no missing values.

# ## Let's do a little bit of EDA!

plt.figure(figsize=(16, 6))

plt.subplot(1, 2, 1)
plt.title("House Price Distribution Plot")
plt.hist(df["price"], bins=25, edgecolor="black", alpha=0.75)
plt.xlabel("House Price (GH₵)")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.boxplot(df["price"], vert=False)
plt.title("House Pricing Spread")
plt.xlabel("House Price (GH₵)")

plt.tight_layout()
plt.show()

print(
    df["price"].describe(
        percentiles=[0.25, 0.50, 0.75, 0.85, 0.90, 1.00]
    )
)

# ### Inference
# 
# The house-price distribution can be inspected for skewness and possible outliers.
# The mean and median should be compared before drawing conclusions about the shape of the distribution.

# ## Visualising Categorical Data

categorical_list = [x for x in df.columns if df[x].dtype == "object"]
print("Categorical columns:")
print(categorical_list)

housing_categorical = [
    "mainroad", "guestroom", "basement",
    "hotwaterheating", "airconditioning", "prefarea"
]

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

for ax, col in zip(axes.ravel(), housing_categorical):
    counts = df[col].value_counts()
    ax.bar(counts.index.astype(str), counts.values)
    ax.set_title(f"{col} Histogram")
    ax.set_xlabel(col)
    ax.set_ylabel("Frequency")
    ax.tick_params(axis="x", rotation=0)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

furnish_counts = df["furnishingstatus"].value_counts()
axes[0].bar(furnish_counts.index.astype(str), furnish_counts.values)
axes[0].set_title("furnishingstatus Histogram")
axes[0].set_xlabel("furnishingstatus")
axes[0].set_ylabel("Frequency")
axes[0].tick_params(axis="x", rotation=30)

loc_counts = df["location"].value_counts().sort_values(ascending=False)
axes[1].bar(loc_counts.index.astype(str), loc_counts.values)
axes[1].set_title("Kumasi Location Frequency")
axes[1].set_xlabel("Kumasi location")
axes[1].set_ylabel("Frequency")
axes[1].tick_params(axis="x", rotation=45)

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Mainroad vs Price")
groups = [df.loc[df["mainroad"] == v, "price"].values for v in sorted(df["mainroad"].unique())]
plt.boxplot(groups, labels=sorted(df["mainroad"].unique()))
plt.xlabel("mainroad")
plt.ylabel("Price (GH₵)")

plt.subplot(1, 2, 2)
plt.title("Guestroom vs Price")
groups = [df.loc[df["guestroom"] == v, "price"].values for v in sorted(df["guestroom"].unique())]
plt.boxplot(groups, labels=sorted(df["guestroom"].unique()))
plt.xlabel("guestroom")
plt.ylabel("Price (GH₵)")

plt.tight_layout()
plt.show()

# **Inference:** The boxplots help compare the distribution of house prices for properties with and without the selected categorical features.

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Basement vs Price")
groups = [df.loc[df["basement"] == v, "price"].values for v in sorted(df["basement"].unique())]
plt.boxplot(groups, labels=sorted(df["basement"].unique()))
plt.xlabel("basement")
plt.ylabel("Price (GH₵)")

plt.subplot(1, 2, 2)
plt.title("Airconditioning vs Price")
groups = [df.loc[df["airconditioning"] == v, "price"].values for v in sorted(df["airconditioning"].unique())]
plt.boxplot(groups, labels=sorted(df["airconditioning"].unique()))
plt.xlabel("airconditioning")
plt.ylabel("Price (GH₵)")

plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.title("Prefarea vs Price")
groups = [df.loc[df["prefarea"] == v, "price"].values for v in sorted(df["prefarea"].unique())]
plt.boxplot(groups, labels=sorted(df["prefarea"].unique()))
plt.xlabel("prefarea")
plt.ylabel("Price (GH₵)")

plt.subplot(1, 2, 2)
plt.title("Furnishing Status vs Price")
furnish_order = sorted(df["furnishingstatus"].unique())
groups = [df.loc[df["furnishingstatus"] == v, "price"].values for v in furnish_order]
plt.boxplot(groups, labels=furnish_order)
plt.xlabel("furnishingstatus")
plt.ylabel("Price (GH₵)")
plt.xticks(rotation=30)

plt.tight_layout()
plt.show()

plt.figure(figsize=(14, 6))
loc_order = df.groupby("location")["price"].median().sort_values(ascending=False).index.tolist()
groups = [df.loc[df["location"] == v, "price"].values for v in loc_order]
plt.boxplot(groups, labels=loc_order)
plt.title("Kumasi Location vs House Price")
plt.xlabel("Kumasi location")
plt.ylabel("Price (GH₵)")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# **Inference:** Location is included because the supplied dataset contains Kumasi neighbourhood information. Differences between neighbourhood price distributions can be examined without assuming that every area has the same effect.

# ## Visualising numerical data

numerical_list = [x for x in df.columns if df[x].dtype in ("int64", "float64")]
print("Numerical columns:")
print(numerical_list)

def scatter_price(x, fig):
    plt.subplot(3, 2, fig)
    plt.scatter(df[x], df["price"], alpha=0.65)
    plt.title(f"{x} vs Price")
    plt.ylabel("Price (GH₵)")
    plt.xlabel(x)

plt.figure(figsize=(12, 14))

scatter_price("area", 1)
scatter_price("bedrooms", 2)
scatter_price("bathrooms", 3)
scatter_price("stories", 4)
scatter_price("parking", 5)

plt.tight_layout()
plt.show()

# **Inference:** Among the numerical variables, `area` is expected to be especially informative because house size has a direct relationship with usable residential space. The scatter plot shows the relationship in this supplied dataset.

from pandas.plotting import scatter_matrix

numeric_for_pairplot = ["price", "area", "bedrooms", "bathrooms", "stories", "parking"]
scatter_matrix(
    df[numeric_for_pairplot],
    figsize=(13, 13),
    diagonal="hist",
    alpha=0.6
)
plt.suptitle("Numerical Variables Pairwise Relationships", y=1.02)
plt.tight_layout()
plt.show()

cor_matrix = df[numerical_list].corr()

plt.figure(figsize=(10, 7))
plt.imshow(cor_matrix, cmap="coolwarm", aspect="auto", vmin=-1, vmax=1)
plt.colorbar(label="Correlation")
plt.xticks(range(len(cor_matrix.columns)), cor_matrix.columns, rotation=45, ha="right")
plt.yticks(range(len(cor_matrix.index)), cor_matrix.index)

for i in range(cor_matrix.shape[0]):
    for j in range(cor_matrix.shape[1]):
        plt.text(j, i, f"{cor_matrix.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=9)

plt.title("Correlation Heatmap")
plt.tight_layout()
plt.show()

# ## Dummy Variables

# Work on a modeling copy.
# city and country contain a single constant value in this dataset,
# so they do not add predictive information and are removed.
df_model = df.drop(columns=["city", "country"]).copy()

categorical_for_model = [
    "mainroad", "guestroom", "basement", "hotwaterheating",
    "airconditioning", "prefarea", "furnishingstatus", "location"
]

df_model = pd.get_dummies(
    df_model,
    columns=categorical_for_model,
    drop_first=True,
    dtype=int
)

df_model.head()

df_model.shape

# The categorical columns are converted to dummy variables so that the machine-learning model can use them as numerical predictors. `location` is included because this is specifically a Kumasi housing dataset.

# ## Train/Test Split

from sklearn.model_selection import train_test_split

np.random.seed(0)

X = df_model.drop("price", axis=1)
y = df_model["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=0.75,
    test_size=0.25,
    random_state=100
)

print("Training rows:", X_train.shape[0])
print("Testing rows:", X_test.shape[0])

# ## Feature Scaling

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

numeric_features = ["area", "bedrooms", "bathrooms", "stories", "parking"]

X_train_scaled = X_train.copy()
X_test_scaled = X_test.copy()

# Fit the scaler on training data only, then apply it to both sets.
X_train_scaled[numeric_features] = scaler.fit_transform(
    X_train[numeric_features]
)
X_test_scaled[numeric_features] = scaler.transform(
    X_test[numeric_features]
)

X_train_scaled.head()

# ## Dividing data into X and y variables

y_train = y_train.copy()
y_test = y_test.copy()

# ## Recursive Feature Elimination (RFE)

from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression

rfe = RFE(
    estimator=LinearRegression(),
    n_features_to_select=10
)
rfe = rfe.fit(X_train_scaled, y_train)

feature_ranking = pd.DataFrame({
    "Feature": X_train_scaled.columns,
    "Selected": rfe.support_,
    "Ranking": rfe.ranking_
}).sort_values(["Selected", "Ranking"], ascending=[False, True])

feature_ranking

selected_features = X_train_scaled.columns[rfe.support_].tolist()
print("Selected features:")
print(selected_features)

X_train_rfe = X_train_scaled[selected_features]
X_train_rfe.head()

# ## Model Building with OLS

import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

def build_model(X, y):
    X_const = sm.add_constant(X, has_constant="add")
    lm = sm.OLS(y, X_const).fit()
    print(lm.summary())
    return X_const

def checkVIF(X):
    vif = pd.DataFrame()
    vif["Features"] = X.columns
    vif["VIF"] = [
        variance_inflation_factor(X.values, i)
        for i in range(X.shape[1])
    ]
    vif["VIF"] = vif["VIF"].replace([np.inf, -np.inf], np.nan)
    vif["VIF"] = vif["VIF"].round(2)
    return vif.sort_values(by="VIF", ascending=False)

# ### MODEL 1

X_train_new = build_model(X_train_rfe, y_train)

# The first OLS model provides coefficient estimates and p-values for the features selected by RFE.

# Check statistical significance before simplifying the model.
model1_pvalues = (
    sm.OLS(y_train, X_train_new).fit()
    .pvalues
    .sort_values()
)
model1_pvalues

# ### Model 2 – retain statistically supported predictors

lm_model2 = sm.OLS(y_train, X_train_new).fit()

# Keep non-constant features with p <= 0.05.
supported_features = [
    feature
    for feature in X_train_rfe.columns
    if lm_model2.pvalues.get(feature, 1.0) <= 0.05
]

if len(supported_features) == 0:
    supported_features = selected_features.copy()

X_train_new = sm.add_constant(
    X_train_scaled[supported_features],
    has_constant="add"
)

lm_model2 = sm.OLS(y_train, X_train_new).fit()
print(lm_model2.summary())

checkVIF(X_train_new)

# ### Model 3 – final model
# 
# The final model is checked for multicollinearity using VIF and retains the feature set that remains suitable for this supplied dataset.

# Iteratively remove only predictors with very high VIF (> 10).
final_features = supported_features.copy()

while len(final_features) > 1:
    X_tmp = sm.add_constant(
        X_train_scaled[final_features],
        has_constant="add"
    )
    vif_tmp = checkVIF(X_tmp).set_index("Features")["VIF"].drop(
        labels=["const"],
        errors="ignore"
    )

    if vif_tmp.empty or vif_tmp.max() <= 10:
        break

    feature_to_drop = vif_tmp.idxmax()
    final_features.remove(feature_to_drop)

X_train_final = sm.add_constant(
    X_train_scaled[final_features],
    has_constant="add"
)

lm = sm.OLS(y_train, X_train_final).fit()
print(lm.summary())

checkVIF(X_train_final)

# ## Model is ready – let's verify the model through Residual Analysis

# Predictions and residuals on the training set
y_train_price = lm.predict(X_train_final)
residuals = y_train - y_train_price

plt.figure(figsize=(10, 5))
plt.hist(residuals, bins=20, edgecolor="black", alpha=0.75)
plt.axvline(0, linestyle="--")
plt.title("Error Terms / Residual Distribution", fontsize=18)
plt.xlabel("Errors (GH₵)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.scatter(y_train_price, residuals, alpha=0.65)
plt.axhline(0, linestyle="--")
plt.title("Residuals vs Predicted Price")
plt.xlabel("Predicted Price (GH₵)")
plt.ylabel("Residual (Actual - Predicted)")
plt.tight_layout()
plt.show()

# **Inference:** The residual plots are used to inspect whether the error terms appear reasonably centered around zero and whether there is an obvious pattern. These are diagnostics rather than proof that every linear-regression assumption is satisfied.

# ## Predicting on the Test Set

X_test_final = sm.add_constant(
    X_test_scaled[final_features],
    has_constant="add"
)

y_pred = lm.predict(X_test_final)

results = pd.DataFrame({
    "Actual Price (GH₵)": y_test.values,
    "Predicted Price (GH₵)": y_pred.values,
    "Absolute Error (GH₵)": np.abs(y_test.values - y_pred.values)
})

results.head(10)

# ## Evaluation of the Model

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred) ** 0.5

evaluation = pd.DataFrame({
    "Metric": ["R-squared (R²)", "MAE", "RMSE"],
    "Value": [r2, mae, rmse]
})

evaluation

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.7)

line_min = min(y_test.min(), y_pred.min())
line_max = max(y_test.max(), y_pred.max())
plt.plot([line_min, line_max], [line_min, line_max], linestyle="--")

plt.title("Actual Price vs Predicted Price", fontsize=18)
plt.xlabel("Actual House Price (GH₵)", fontsize=14)
plt.ylabel("Predicted House Price (GH₵)", fontsize=14)
plt.tight_layout()
plt.show()

# ## Example Prediction
# 
# The example below shows how the trained model can be used for a new Kumasi house. The property information is only an example input for demonstrating the prediction code.

# Example new property
example_house = {
    "area": 5500,
    "bedrooms": 3,
    "bathrooms": 2,
    "stories": 2,
    "parking": 1,
    "mainroad": "yes",
    "guestroom": "no",
    "basement": "no",
    "hotwaterheating": "no",
    "airconditioning": "yes",
    "prefarea": "yes",
    "furnishingstatus": "semi-furnished",
    "location": "Ahodwo"
}

example_df = pd.DataFrame([example_house])

# Create dummy columns using the same training structure.
example_encoded = pd.get_dummies(
    example_df,
    columns=categorical_for_model,
    drop_first=True,
    dtype=int
)

example_encoded = example_encoded.reindex(
    columns=X.columns,
    fill_value=0
)

example_scaled = example_encoded.copy()
example_scaled[numeric_features] = scaler.transform(
    example_encoded[numeric_features]
)

example_final = sm.add_constant(
    example_scaled[final_features],
    has_constant="add"
)

predicted_price = float(lm.predict(example_final).iloc[0])

print(f"Estimated example house price: GH₵{predicted_price:,.2f}")

# ## Final Model Summary
# 
# The final section reports the selected features, test-set evaluation metrics, and the fitted OLS model summary. The model should be evaluated using the supplied dataset's limitations in mind.

print("Final selected features:")
for feature in final_features:
    print(" -", feature)

print("\nTest-set metrics")
print(f"R²   : {r2:.4f}")
print(f"MAE  : GH₵{mae:,.2f}")
print(f"RMSE : GH₵{rmse:,.2f}")

print("\nOLS Model Summary")
print(lm.summary())

# ### Conclusion
# 
# This notebook provides a complete machine-learning workflow for **Kumasi Ghana house-price prediction** using the supplied CSV dataset. It follows the structure of the referenced regression notebook while adding the Kumasi `location` field and an example prediction for a new property.

import joblib

model_bundle = {
    "model": lm,
    "scaler": scaler,
    "numeric_features": numeric_features,
    "categorical_for_model": categorical_for_model,
    "model_columns": X.columns.tolist(),
    "final_features": final_features,
    "target": "price",
    "currency": "GHS (GH₵)",
    "city": "Kumasi",
    "country": "Ghana"
}

MODEL_PATH = Path("kumasi_house_price_model.pkl")
joblib.dump(model_bundle, MODEL_PATH)

print(f"Saved trained model bundle to: {MODEL_PATH.resolve()}")
