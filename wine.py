import pandas as pd
from sklearn.datasets import load_wine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Load wine dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
# The above code snippet was generated using ChatGPT 5.0 on 11/10/25 at 1:05p.
y=wine.target

# Finding column names
print(wine.feature_names)
print(wine.target_names)

# Building pipeline
pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),      
    ('scaler', StandardScaler()),                     
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('clf', RandomForestClassifier(n_estimators=100, random_state=42))
])
# Using EdStem lessons from AIPI503

# Splitting data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)

# Using pipeline and training model
pipeline.fit(X_train, y_train)
y_pred = pipeline.predict(X_test)
# The above code snippet was generated using ChatGPT 5.0 on 11/10/25 at 1:14p.

accuracy = accuracy_score(y_test, y_pred, normalize=True, sample_weight=None)

print(accuracy)
