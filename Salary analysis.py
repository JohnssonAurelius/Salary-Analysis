import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
import matplotlib.pyplot as plt
from sklearn.tree import export_graphviz
import graphviz

df = pd.read_csv('data.csv')
df_edu = pd.read_csv("education_data.csv")

df.shape

#Checking all data
df_edu

df

df.shape

df_edu.shape

#Merging both dataset 
merged_df = pd.merge(df, df_edu, on="unique_id", how="inner")

#Check if the merging works 
merged_df.shape

merged_df.isnull().sum()

#Checking the distribution of income
plt.figure(figsize=(20, 10))
sns.histplot(merged_df['income'], kde=True, bins=80)
plt.title('Distribution of income')
plt.show()

#Checking for missing data
merged_df.isna().sum()

merged_df.info()

merged_df.isna().sum()

#There are some data with "?" that were not included as NA, replacing all "?" with NA
merged_df = merged_df.replace(" ?", pd.NA)

#Checking that data with "?" is being noticed as NA
merged_df.isna().sum()

merged_df

merged_df.groupby("race")["native-country"].count()

new_df = merged_df.dropna(subset=["native-country"])

#Checking if the "native-country" imputation worked
new_df.isnull().sum()

new_df.shape

#to avoid any sex bias, filled all NA sex as "Unknown"
new_df["sex"] = new_df["sex"].fillna("Unknown")

#Checking if it worked
new_df.isna().sum()

new_df

# Impute workclass within each education group
new_df['workclass'] = (
    new_df.groupby(['education'], group_keys=False)['workclass']
    .apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
)


# Impute occupation within each workclass × education group
new_df['occupation'] = (
    new_df.groupby(['workclass', 'education'], group_keys=False)['occupation']
    .apply(lambda x: x.fillna(x.mode()[0] if not x.mode().empty else "Unknown"))
)

#Checking if the imputation worked
new_df.isnull().sum()

#Checking the new imputed data 
new_df

for col in ["workclass", "occupation"]:
    print(f"\nValue counts for {col}:")
    print(merged_df[col].value_counts(dropna=False).head(20))

#Checking value counts for "workclass" and "occupation"
for col in ["workclass", "occupation"]:
    print(f"\nValue counts for {col}:")
    print(new_df[col].value_counts(dropna=False).head(20))

plt.figure(figsize=(20, 10))
sns.histplot(new_df['income'], kde=True, bins=80)
plt.title('Distribution of income')
plt.show()

#Education income analysis
order1 = new_df.groupby("education-num")["income"].mean().sort_values().index

plt.figure(figsize=(10, 6))
sns.barplot(x='education-num', y='income', data=new_df, order=order1, color="green")
plt.title("income distribution on education")
plt.xlabel('education')
plt.ylabel('income')
plt.show()

#Checking if there are any outliers
plt.figure(figsize=(10, 6))
plt.figure(figsize=(10, 6))
sns.boxplot(x='education-num', y='income', data=new_df)
plt.title('education vs income')
plt.show()

#Occupation income analysis
order2 = new_df.groupby("occupation")["income"].mean().sort_values().index
plt.figure(figsize=(20, 12))
sns.barplot(x='occupation', y='income', data=new_df, color="black", order=order2)
plt.title("income distribution on Occupation")
plt.xlabel('occupation')
plt.ylabel('income')
plt.show()

#Hours worked and occupation to income analysis
avg_income = new_df.groupby(["occupation","hours-per-week"])["income"].mean().reset_index()

avg_income.sort_values("income", ascending=False).head(10)


#Scatterplot for hours worked analysis
plt.figure(figsize=(10,6))
sns.scatterplot(x="hours-per-week", y="income", data=new_df,
                alpha=0.9, s=20)
plt.title("Income vs Hours per Week")
plt.show()

#Line graph for hours worked analysis
sns.lineplot(x="hours-per-week", y="income", data=avg_income)
plt.title("Average Income vs Hours per Week")
plt.show()


# Select numerical columns
numerical_columns = new_df.select_dtypes(include=[np.number]).columns

# Compute correlation matrix
correlation_matrix = new_df[numerical_columns].corr()

# Visualize correlation matrix
plt.figure(figsize=(18, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

avg_metrics = new_df.groupby('education-num').agg({
    'income': 'mean',
    'hours-per-week': 'mean'
}).round(2)

print("\nAverage metrics by education:")
print(avg_metrics)

fig, axes = plt.subplots(2, 1, figsize=(12,10))

sns.barplot(x='marital-status', y='income', data=new_df, 
            color="turquoise", ax=axes[0])
axes[0].set_title("Income by Marital Status")
axes[0].set_xlabel("Marital Status"); axes[0].set_ylabel("Income")
axes[0].tick_params(axis='x', rotation=45)

order = new_df.groupby("workclass")["income"].mean().sort_values().index
sns.barplot(x='workclass', y='income', data=new_df, 
            order=order, color="blue", ax=axes[1])
axes[1].set_title("Income by Workclass")
axes[1].set_xlabel("Workclass"); axes[1].set_ylabel("Income")
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()


# Select features for prediction
features = ['workclass', 'hours-per-week', 'education-num', "occupation", "marital-status", "sex", "birth_year", "relationship", "race"]

# Create a new dataframe with only the selected features and price
df_selected = new_df[features + ['income']]


# Convert categorical variables to dummy variables
df_encoded = pd.get_dummies(df_selected, columns=['workclass', "occupation", "marital-status", "sex", "birth_year", "relationship", "race"])

# Separate features (X) and target variable (y)
X = df_encoded.drop('income', axis=1)
y = df_encoded['income']


#Splitting the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#Linear regression model 
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_predictions = lr_model.predict(X_test)

#Random forest model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_predictions = rf_model.predict(X_test)

plt.figure(figsize=(8, 5))
plt.scatter(y_test, lr_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title("Linear Regression: Actual vs Predicted")

plt.tight_layout()
plt.show()

# Visualize predictions vs actual
plt.figure(figsize=(8, 5))
plt.scatter(y_test, rf_predictions, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.xlabel("Actual Income")
plt.ylabel("Predicted Income")
plt.title("Random Forest: Actual vs Predicted")

plt.tight_layout()
plt.show()

#Evaluations of the models built
def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)

    print(f"{model_name} Results:")
    print(f"Mean Absolute Error (MAE): {mae:.2f}")
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
    print(f"Mean Absolute Percentage Error (MAPE): {mape:.2%}")
    print("-"*50)

evaluate_model(y_test, rf_predictions, "Random Forest")
evaluate_model(y_test, lr_predictions, "Linear Regression")


rf_importances = pd.Series(
    rf_model.feature_importances_, 
    index=X_train.columns
).sort_values(ascending=False)

# Plot top 15
plt.figure(figsize=(10,6))
rf_importances.head(15).plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Random Forest - Feature Importance")
plt.xlabel("Importance Score")
plt.show()

