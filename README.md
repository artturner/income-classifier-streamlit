# Income Prediction Streamlit App

This Streamlit application predicts whether a person's income is above or below $50,000 based on various demographic and employment attributes using an XGBoost model.

## Features

- **Interactive Web Interface**: Easy-to-use form for inputting personal attributes
- **Real-time Predictions**: Get instant income predictions with confidence scores
- **High Accuracy Model**: Uses XGBoost model with 87.04% accuracy and 92.28% ROC-AUC
- **Probability Breakdown**: Shows detailed probability for each income class

## Model Performance

- **Accuracy**: 87.04%
- **Precision**: 78.33%
- **Recall**: 63.84%
- **F1-Score**: 70.34%
- **ROC-AUC**: 92.28%

## Deployment

### Local Development

1. Clone this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Copy the model files to the `models/` directory:
   - `xgboost_final.pkl`
   - `preprocessor.pkl`
4. Run the app:
   ```bash
   streamlit run app.py
   ```

### Streamlit Community Cloud

1. Push this repository to GitHub
2. Connect your GitHub account to Streamlit Community Cloud
3. Deploy the app by selecting this repository
4. Ensure model files are included in the repository or uploaded as secrets

## Input Features

The app requires the following inputs:

### Personal Information
- Age (17-90)
- Sex (Male/Female)
- Race (White, Black, Asian-Pac-Islander, etc.)

### Education
- Education Level (Bachelors, Some-college, HS-grad, etc.)
- Education Years (1-16)

### Work Information
- Work Class (Private, Self-emp-not-inc, etc.)
- Occupation (Tech-support, Sales, Exec-managerial, etc.)
- Hours per Week (1-99)

### Family Information
- Marital Status (Married-civ-spouse, Divorced, etc.)
- Relationship (Wife, Husband, Own-child, etc.)

### Financial Information
- Capital Gain (0-99999)
- Capital Loss (0-4356)

### Location
- Native Country (United-States, Mexico, etc.)
- Final Weight (demographic weight)

## File Structure

```
streamlit_app/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── README.md          # This file
└── models/            # Model files (to be copied)
    ├── xgboost_final.pkl
    └── preprocessor.pkl
```

## License

This project is for educational purposes as part of the SDS-CP034 Income Insight analysis.