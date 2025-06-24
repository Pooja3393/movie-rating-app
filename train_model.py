import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib

# Load data
df = pd.read_csv("imdb_top_1000.csv")

# Take first genre only
df['Genre'] = df['Genre'].apply(lambda x: str(x).split(',')[0] if pd.notna(x) else 'Unknown')

# Drop rows with missing values in important columns
df = df.dropna(subset=['Genre', 'Director', 'Actor 1', 'Rating'])

# Features and target
X = df[['Genre', 'Director', 'Actor 1']]
y = df['Rating']

# Preprocessing pipeline for categorical features
categorical_features = ['Genre', 'Director', 'Actor 1']
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', categorical_transformer, categorical_features)
    ])

# Create pipeline: preprocessing + model
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(n_estimators=10, random_state=42))

])

# Train model
model.fit(X, y)

# Save model as 'model.pkl'
joblib.dump(model, 'model.pkl')

print("Model training done and saved as model.pkl")