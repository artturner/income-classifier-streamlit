# Deployment Guide

## Streamlit Community Cloud Deployment

### Prerequisites
1. GitHub account
2. Streamlit Community Cloud account (free at https://share.streamlit.io)

### Steps

1. **Create a new GitHub repository**
   ```bash
   # Create a new repository on GitHub (e.g., "income-prediction-app")
   ```

2. **Upload the app files**
   - Upload all files from the `streamlit_app/` directory to your new repository
   - Make sure the following files are included:
     - `app.py` (main application)
     - `requirements.txt` (dependencies)
     - `models/xgboost_final.pkl` (trained model)
     - `models/preprocessor.pkl` (data preprocessor)
     - `README.md` (documentation)

3. **Deploy to Streamlit Community Cloud**
   - Go to https://share.streamlit.io
   - Sign in with your GitHub account
   - Click "Create app"
   - Select your repository
   - Set the main file path to `app.py`
   - Click "Deploy"

4. **Verify deployment**
   - Wait for the app to build and deploy (usually takes 2-5 minutes)
   - Test the app with sample inputs
   - Share the public URL with users

### Alternative: Local Development

```bash
# Clone the repository
git clone <your-repo-url>
cd income-prediction-app

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

### Troubleshooting

**Common Issues:**
1. **Import errors**: Ensure all dependencies are in `requirements.txt`
2. **Model loading errors**: Verify model files are in the correct `models/` directory
3. **Memory issues**: The free tier has limited resources; consider model optimization if needed

**File Size Limits:**
- Individual files: 25MB max
- Repository size: 1GB max
- If models are too large, consider using Git LFS or external storage

### Configuration

The app uses the following default settings:
- Port: 8501 (for local development)
- Model: XGBoost (87.04% accuracy)
- UI: Wide layout with form-based input

These can be modified in `app.py` if needed.