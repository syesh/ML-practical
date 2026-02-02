import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
data = fetch_california_housing(as_frame=True)
housing_df=data.frame
numerical_features=housing_df.select_dtypes(include=[np.number])
numerical_features.head()

plt.figure(figsize=(15,10))
for i,feature in enumerate(numerical_features.columns):
    plt.subplot(3,3,i+1)
    sns.histplot(data=housing_df[feature],kde=True,bins=45,color='blue')
    plt.title(f'Histogram of {feature}')
plt.tight_layout()
plt.show()

plt.figure(figsize=(15,10))
for i,feature in enumerate(numerical_features.columns):
    plt.subplot(3,3,i+1)
    sns.boxplot(data=housing_df[feature],color='orange')
    plt.title(f'Box plot of {feature}')
plt.tight_layout()
plt.show()

print("Description of outliers")
outliers_summary={}
for feature in numerical_features.columns:
    Q1=housing_df[feature].quantile(0.25)
    Q3=housing_df[feature].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound=Q1-1.5*IQR
    upper_bound=Q3+1.5*IQR
    outliers=housing_df[(housing_df[feature]<lower_bound) | (housing_df[feature]>upper_bound)]
    outliers_summary[feature] = len(outliers)
print(outliers_summary)
print(housing_df.describe())
