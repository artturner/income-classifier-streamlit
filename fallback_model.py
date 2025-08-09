"""
Fallback Model Creation
======================
Creates a simple but effective model when the original pickle files can't be loaded
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
import joblib

def create_fallback_model():
    """Create and train a fallback model"""
    
    # Load the original data
    try:
        data = pd.read_csv('../adult.csv')
    except:
        # If adult.csv is not available, create a dummy model
        return create_dummy_model()
    
    # Basic preprocessing
    # Handle missing values
    data = data.replace('?', np.nan)
    data = data.dropna()
    
    # Separate features and target
    X = data.drop('income', axis=1)
    y = data['income'].apply(lambda x: 1 if x.strip() == '>50K' else 0)
    
    # Identify categorical and numerical columns
    categorical_features = ['workclass', 'education', 'marital.status', 'occupation',
                           'relationship', 'race', 'sex', 'native.country']
    numerical_features = ['age', 'fnlwgt', 'education.num', 'capital.gain', 
                         'capital.loss', 'hours.per.week']
    
    # Create preprocessor
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ])
    
    # Fit preprocessor
    X_processed = preprocessor.fit_transform(X)
    
    # Train a simple Random Forest model
    model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_processed, y)
    
    return model, preprocessor

def create_dummy_model():
    """Create a dummy model for demonstration purposes"""
    from sklearn.dummy import DummyClassifier
    from sklearn.preprocessing import StandardScaler
    
    # Create a dummy classifier that predicts based on prior
    model = DummyClassifier(strategy='prior', random_state=42)
    
    # Create a dummy preprocessor
    preprocessor = StandardScaler()
    
    # Fit with dummy data
    dummy_X = np.random.random((100, 14))
    dummy_y = np.random.choice([0, 1], 100)
    
    model.fit(dummy_X, dummy_y)
    preprocessor.fit(dummy_X)
    
    return model, preprocessor

if __name__ == "__main__":
    model, preprocessor = create_fallback_model()
    
    # Save the fallback model
    joblib.dump(model, 'models/fallback_model.joblib')
    joblib.dump(preprocessor, 'models/fallback_preprocessor.joblib')
    
    print("Fallback model created and saved!")