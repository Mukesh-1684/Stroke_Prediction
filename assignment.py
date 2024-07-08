import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, MinMaxScaler, StandardScaler
from sklearn.impute import SimpleImputer
import warnings
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Ignore all future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# Dataset importing 
data = pd.read_csv("healthcare.csv")

# Dataframe conversion
df = pd.DataFrame(data)
print(df.head())

# Display null count in dataset
print("Null count present in dataset \n", df.isnull().sum())

print("*****DATA PRE-PROCESSING STEPS*****")

# Step 1: Data cleaning

# Filling null data with specific values
df["hypertension"] = df["hypertension"].fillna(1)

# Filling null values using mean, median, and mode
df["age"].fillna(df["age"].mean(), inplace=True)
df["bmi"].fillna(df["bmi"].median(), inplace=True)

# Filling null values using SimpleImputer
imputer = SimpleImputer(strategy='mean')
df["avg_glucose_level"] = imputer.fit_transform(df[["avg_glucose_level"]])
df["heart_disease"] = imputer.fit_transform(df[["heart_disease"]])

# Step 2: Encoding

# Filling string type null values using LabelEncoder
label_encoder = LabelEncoder()
df['ever_married'] = label_encoder.fit_transform(df["ever_married"])
df["ever_married"].fillna(df["ever_married"].median(), inplace=True)
df['Residence_type'] = label_encoder.fit_transform(df["Residence_type"])
df["Residence_type"].fillna(df["Residence_type"].median(), inplace=True)

# Filling string type null values using OneHotEncoder
one_hot_encoder = OneHotEncoder(sparse_output=False)
work_type_encoded = one_hot_encoder.fit_transform(df[["work_type"]])
df = df.join(pd.DataFrame(work_type_encoded, columns=one_hot_encoder.get_feature_names_out(['work_type'])))

gender_encoded = one_hot_encoder.fit_transform(df[["gender"]])
df = df.join(pd.DataFrame(gender_encoded, columns=one_hot_encoder.get_feature_names_out(['gender'])))

smoking_status_encoded = one_hot_encoder.fit_transform(df[["smoking_status"]])
df = df.join(pd.DataFrame(smoking_status_encoded, columns=one_hot_encoder.get_feature_names_out(['smoking_status'])))

# Drop the original categorical columns after encoding
df.drop(columns=['work_type', 'gender', 'smoking_status'], inplace=True)

# Display null count after filling values
print("After filling all the null values\n", df.isnull().sum())

# Step 3: Normalization
numeric_columns = df.select_dtypes(include=[np.number])
min_max_scaler = MinMaxScaler()
scaled_min_max = min_max_scaler.fit_transform(numeric_columns)
scaled_df = pd.DataFrame(scaled_min_max, columns=numeric_columns.columns)

# Step 4: Feature Scaling
standard_scaler = StandardScaler()
scaled_standard = standard_scaler.fit_transform(numeric_columns)
standard_df = pd.DataFrame(scaled_standard, columns=numeric_columns.columns)

print("After data preprocessing,\n", df.isnull().sum())

print("***************Exploratory Data Analysis**********")
heart_disease_groups = df.groupby('heart_disease')['stroke'].mean()
glucose_groups = df.groupby(pd.cut(df['avg_glucose_level'], bins=[0, 70, 100, 140, 200]))['stroke'].mean()
hypertension_groups = df.groupby('hypertension')['stroke'].mean()

# Plotting
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Heart Disease Bar Plot
heart_disease_groups.plot(kind='bar', ax=axes[0], color='skyblue')
axes[0].set_title('Stroke Probability by Heart Disease')
axes[0].set_xlabel('Heart Disease')
axes[0].set_ylabel('Stroke Probability')
axes[0].set_xticks([0, 1])
axes[0].set_xticklabels(['No', 'Yes'], rotation=0)

# Glucose Level Bar Plot
glucose_groups.plot(kind='bar', ax=axes[1], color='lightgreen')
axes[1].set_title('Stroke Probability by Avg Glucose Level')
axes[1].set_xlabel('Glucose Level Range')
axes[1].set_ylabel('Stroke Probability')

# Hypertension Bar Plot
hypertension_groups.plot(kind='bar', ax=axes[2], color='salmon')
axes[2].set_title('Stroke Probability by Hypertension')
axes[2].set_xlabel('Hypertension')
axes[2].set_ylabel('Stroke Probability')
axes[2].set_xticks([0, 1])
axes[2].set_xticklabels(['No', 'Yes'], rotation=0)



print("****************Feature Extraction*************")
important_features = ['hypertension', 'smoking_status_Unknown', 'smoking_status_formerly smoked', 'smoking_status_never smoked', 'smoking_status_smokes', 'avg_glucose_level', 'heart_disease']
x = df[important_features]
y = df["stroke"]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
pca = PCA(n_components=2)
x_pca = pca.fit_transform(x_scaled)
principal_components = pca.components_
print(principal_components)

print("****************Model Building****************")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
linear_model = LinearRegression()
linear_model.fit(x_train, y_train)
y_prediction = linear_model.predict(x_test)
print("*************** y_prediction")
print("Prediction result \n", y_prediction)
print("Coefficients:", linear_model.coef_)

print("************Evaluation Metrics*******************")
mean_square_error = mean_squared_error(y_test, y_prediction)
print("Mean Squared Error:", mean_square_error)

r2_score_value = r2_score(y_test, y_prediction)
print("R2 Score:", r2_score_value)

# Correlation Matrix
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.show()

# Pair Plots
sns.pairplot(df, hue='stroke', vars=['age', 'bmi', 'avg_glucose_level', 'hypertension'])
plt.show()

# Cross-Validation and Hyperparameter Tuning (Example for Random Forest)
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

rf = RandomForestClassifier()
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_features': ['auto', 'sqrt'],
    'max_depth': [10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

print("Best parameters found: ", grid_search.best_params_)
print("Best cross-validation score: {:.2f}".format(grid_search.best_score_))

# Model Evaluation with the best estimator
best_rf = grid_search.best_estimator_
y_pred_rf = best_rf.predict(x_test)
print("Random Forest MSE:", mean_squared_error(y_test, y_pred_rf))
print("Random Forest R2 Score:", r2_score(y_test, y_pred_rf))

plt.tight_layout()
plt.show()  
