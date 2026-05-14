from sklearn.datasets import load_iris, load_diabetes, fetch_openml
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt

#irisData = load_iris()
diabetesData = load_diabetes()
#openmlData = fetch_openml()

#print(irisData.keys())

#irisDataFrame = pd.DataFrame( 
   # data=irisData.data,
   # columns=irisData.feature_names)
#print(irisDataFrame)

diabetesDataFrame = pd.DataFrame(
    data = diabetesData.data,
    columns = diabetesData.feature_names
)

print(diabetesDataFrame[0:20])

# On Diabetes Data
#basic EDA (summary stats, missing values, class distribution)

print(diabetesDataFrame.head(20))
print(diabetesDataFrame.describe())
diabetesDataFrame.info()
print(diabetesDataFrame.isnull().sum())


#Train-Test Split
y = diabetesData.target
print(pd.Series(y).describe())
X = diabetesDataFrame
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train size:", X_train.shape)
print("Test  size:", X_test.shape)

#feature scaling 
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)   # fit + transform on train
X_test_scaled  = scaler.transform(X_test)         # only transform on test

print("Scaled train sample:\n", X_train_scaled[:3])

#correlation matrix
corr_matrix = diabetesDataFrame.corr()
plt.figure(figsize=(10, 8))

#plot
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm")
plt.title("Feature Diabetes Correlation Heatmap")
plt.tight_layout()
plt.show()



