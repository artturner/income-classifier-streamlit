#!/usr/bin/env python3
"""
Model Conversion Script
======================
Converts pickle models to joblib format for better cross-platform compatibility
"""

import pickle
import joblib
import os

def convert_models():
    """Convert pickle models to joblib format"""
    
    # Model files to convert
    model_files = {
        'xgboost_final.pkl': 'xgboost_final.joblib',
        'preprocessor.pkl': 'preprocessor.joblib'
    }
    
    for pickle_file, joblib_file in model_files.items():
        pickle_path = f'models/{pickle_file}'
        joblib_path = f'models/{joblib_file}'
        
        if os.path.exists(pickle_path):
            try:
                # Load with pickle
                with open(pickle_path, 'rb') as f:
                    model = pickle.load(f)
                
                # Save with joblib
                joblib.dump(model, joblib_path, compress=3)
                print(f"✓ Converted {pickle_file} → {joblib_file}")
                
            except Exception as e:
                print(f"✗ Failed to convert {pickle_file}: {str(e)}")
        else:
            print(f"✗ File not found: {pickle_path}")

if __name__ == "__main__":
    convert_models()