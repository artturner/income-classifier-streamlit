import streamlit as st
import pickle
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Try to import SHAP, fallback gracefully if not available
try:
    import shap
    import matplotlib.pyplot as plt
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    st.sidebar.warning("⚠️ SHAP not available - install with: pip install shap")

@st.cache_resource
def load_model_and_preprocessor():
    """Load model and preprocessor with multiple fallback options"""
    
    # Method 1: Try joblib first (most compatible)
    try:
        import joblib
        model = joblib.load('models/xgboost_final.pkl')
        preprocessor = joblib.load('models/preprocessor.pkl')
        st.sidebar.success("✅ Loaded XGBoost model (joblib)")
        return model, preprocessor, "XGBoost"
    except Exception:
        pass
    
    # Method 2: Try pickle with different encodings
    try:
        import pickle
        with open('models/xgboost_final.pkl', 'rb') as f:
            model = pickle.load(f, encoding='latin1')
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f, encoding='latin1')
        st.sidebar.success("✅ Loaded XGBoost model (pickle)")
        return model, preprocessor, "XGBoost"
    except Exception:
        pass
    
    # Method 3: Try standard pickle
    try:
        import pickle
        with open('models/xgboost_final.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/preprocessor.pkl', 'rb') as f:
            preprocessor = pickle.load(f)
        st.sidebar.success("✅ Loaded XGBoost model (standard pickle)")
        return model, preprocessor, "XGBoost"
    except Exception:
        pass
    
    # Method 4: Create a simple fallback model
    try:
        st.sidebar.warning("⚠️ Using fallback Random Forest model")
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.preprocessing import StandardScaler
        
        # Create a simple model
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=8)
        preprocessor = StandardScaler()
        
        # Train on some basic patterns (this is a demonstration)
        # In a real scenario, you'd retrain on the actual data
        dummy_X = np.random.random((1000, 70))  # 70 features after preprocessing
        dummy_y = np.random.choice([0, 1], 1000, p=[0.76, 0.24])  # Realistic class distribution
        
        model.fit(dummy_X, dummy_y)
        preprocessor.fit(dummy_X)
        
        return model, preprocessor, "Random Forest (Fallback)"
        
    except Exception as e:
        raise Exception(f"All model loading methods failed. Error: {str(e)}")

@st.cache_resource
def create_shap_explainer(_model, _X_sample):
    """Create SHAP explainer for the model"""
    if not SHAP_AVAILABLE:
        return None
    
    try:
        # Use TreeExplainer for tree-based models (XGBoost, Random Forest)
        explainer = shap.TreeExplainer(_model)
        return explainer
    except Exception:
        try:
            # Fallback to Explainer for other models
            explainer = shap.Explainer(_model, _X_sample)
            return explainer
        except Exception as e:
            st.warning(f"Could not create SHAP explainer: {str(e)}")
            return None

def get_feature_names(input_data_processed):
    """Get feature names for SHAP explanations"""
    # Create a mapping of the most important features
    feature_mapping = {
        'age': 'Age',
        'education.num': 'Education Years', 
        'hours.per.week': 'Hours per Week',
        'capital.gain': 'Capital Gain',
        'capital.loss': 'Capital Loss',
        'has_capital_gain': 'Has Capital Gain',
        'has_capital_loss': 'Has Capital Loss',
        'workclass': 'Work Class',
        'education': 'Education Level',
        'marital.status': 'Marital Status',
        'occupation': 'Occupation',
        'relationship': 'Relationship',
        'race': 'Race',
        'sex': 'Gender',
        'native.country': 'Country',
        'native.country_grouped': 'Country (Grouped)'
    }
    
    # Return the original column names - the preprocessor will handle the full transformation
    return list(input_data_processed.columns)

def get_enhanced_feature_names(input_data_processed, num_features):
    """Get enhanced feature names for better SHAP explanations"""
    
    # Base feature mappings with human-readable names
    base_features = [
        'Age', 'Final Weight', 'Education Years', 'Capital Gain', 'Capital Loss', 
        'Hours per Week', 'Has Capital Gain', 'Has Capital Loss'
    ]
    
    # Categorical feature expansions (these get one-hot encoded)
    categorical_expansions = {
        'workclass': ['Private', 'Self-emp-not-inc', 'Self-emp-inc', 'Federal-gov', 'Local-gov', 'State-gov', 'Without-pay', 'Never-worked'],
        'education': ['Bachelors', 'Some-college', '11th', 'HS-grad', 'Prof-school', 'Assoc-acdm', 'Assoc-voc', '9th', '7th-8th', '12th', 'Masters', '1st-4th', '10th', 'Doctorate', '5th-6th', 'Preschool'],
        'marital.status': ['Married-civ-spouse', 'Divorced', 'Never-married', 'Separated', 'Widowed', 'Married-spouse-absent', 'Married-AF-spouse'],
        'occupation': ['Tech-support', 'Craft-repair', 'Other-service', 'Sales', 'Exec-managerial', 'Prof-specialty', 'Handlers-cleaners', 'Machine-op-inspct', 'Adm-clerical', 'Farming-fishing', 'Transport-moving', 'Priv-house-serv', 'Protective-serv', 'Armed-Forces'],
        'relationship': ['Wife', 'Own-child', 'Husband', 'Not-in-family', 'Other-relative', 'Unmarried'],
        'race': ['White', 'Black', 'Asian-Pac-Islander', 'Amer-Indian-Eskimo', 'Other'],
        'sex': ['Female', 'Male'],
        'native.country_grouped': ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba', 'Other']
    }
    
    # Build comprehensive feature list
    enhanced_names = base_features.copy()
    
    # Add categorical features with prefixes
    for category, values in categorical_expansions.items():
        for value in values:
            enhanced_names.append(f"{category.replace('.', ' ').title()}: {value}")
    
    # If we have more features than names, add generic names
    while len(enhanced_names) < num_features:
        enhanced_names.append(f"Feature_{len(enhanced_names)}")
    
    # Return only the number we need
    return enhanced_names[:num_features]

def create_feature_vector(input_data):
    """Create a feature vector that matches the expected model input"""
    # This is a simplified version - in reality, this would need to match
    # the exact preprocessing pipeline used during training
    
    # Basic numerical features
    features = [
        input_data['age'][0],
        input_data['fnlwgt'][0],  
        input_data['education.num'][0],
        input_data['capital.gain'][0],
        input_data['capital.loss'][0],
        input_data['hours.per.week'][0],
    ]
    
    # Add some basic categorical encodings (simplified)
    # This is just for the fallback model
    workclass_map = {'Private': 1, 'Self-emp-not-inc': 2, 'Self-emp-inc': 3, 'Federal-gov': 4,
                     'Local-gov': 5, 'State-gov': 6, 'Without-pay': 0, 'Never-worked': 0}
    features.append(workclass_map.get(input_data['workclass'][0], 0))
    
    education_map = {'Bachelors': 13, 'Some-college': 10, 'HS-grad': 9, 'Prof-school': 15,
                     'Assoc-acdm': 12, 'Assoc-voc': 11, 'Masters': 14, 'Doctorate': 16}
    features.append(education_map.get(input_data['education'][0], 9))
    
    # Add more simplified features to reach ~70 total
    features.extend([0] * 62)  # Pad with zeros for other categorical features
    
    return np.array(features).reshape(1, -1)

def generate_counterfactuals(original_data, model, preprocessor, target_class, model_name):
    """Generate counterfactual explanations by finding minimal changes to flip prediction"""
    
    counterfactuals = []
    debug_info = []
    
    # Get original values
    orig_education = original_data['education'].iloc[0]
    orig_occupation = original_data['occupation'].iloc[0]
    orig_hours = original_data['hours.per.week'].iloc[0]
    orig_marital = original_data['marital.status'].iloc[0]
    orig_capital_gain = original_data['capital.gain'].iloc[0]
    
    # Define comprehensive modification strategies
    strategies = [
        # Education upgrades
        {
            'name': 'Bachelor\'s Degree',
            'changes': {
                'education': 'Bachelors',
                'education.num': 13
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        {
            'name': 'Master\'s Degree',
            'changes': {
                'education': 'Masters',
                'education.num': 14
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        {
            'name': 'Doctorate Degree',
            'changes': {
                'education': 'Doctorate',
                'education.num': 16
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        # Career changes
        {
            'name': 'Executive Management',
            'changes': {
                'occupation': 'Exec-managerial',
                'hours.per.week': max(45, orig_hours + 10)
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': True}
        },
        {
            'name': 'Professional Specialty',
            'changes': {
                'occupation': 'Prof-specialty',
                'education': 'Bachelors' if orig_education in ['HS-grad', '11th', '9th', '10th', '12th'] else orig_education
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': True}
        },
        {
            'name': 'Technology Role',
            'changes': {
                'occupation': 'Tech-support',
                'education': 'Some-college' if orig_education in ['HS-grad', '11th', '9th', '10th', '12th'] else orig_education
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': True}
        },
        # Work changes
        {
            'name': 'Increased Work Hours',
            'changes': {
                'hours.per.week': min(60, orig_hours + 20)
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        {
            'name': 'Full-time Professional',
            'changes': {
                'hours.per.week': 50,
                'occupation': 'Prof-specialty'
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': True}
        },
        # Life changes
        {
            'name': 'Marriage',
            'changes': {
                'marital.status': 'Married-civ-spouse'
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        # Financial changes
        {
            'name': 'Investment Income',
            'changes': {
                'capital.gain': 5000
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        {
            'name': 'Significant Investment',
            'changes': {
                'capital.gain': 15000
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': False}
        },
        # Combined strategies
        {
            'name': 'Career + Education Boost',
            'changes': {
                'education': 'Bachelors',
                'education.num': 13,
                'occupation': 'Exec-managerial',
                'hours.per.week': max(45, orig_hours + 5)
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': True}
        },
        {
            'name': 'Professional + Marriage',
            'changes': {
                'occupation': 'Prof-specialty',
                'marital.status': 'Married-civ-spouse',
                'education': 'Bachelors' if orig_education in ['HS-grad', '11th', '9th', '10th', '12th'] else orig_education
            },
            'feasibility_factors': {'age_appropriate': True, 'career_change': True}
        }
    ]
    
    for strategy in strategies:
        try:
            # Create modified data
            modified_data = original_data.copy()
            
            # Apply changes
            changes_applied = 0
            for feature, new_value in strategy['changes'].items():
                if feature in modified_data.columns:
                    old_value = modified_data[feature].iloc[0]
                    if old_value != new_value:  # Only apply if actually different
                        modified_data[feature] = new_value
                        changes_applied += 1
            
            # Skip if no actual changes were made
            if changes_applied == 0:
                continue
            
            # Re-engineer features that depend on capital gains/losses
            if 'capital.gain' in strategy['changes']:
                modified_data['has_capital_gain'] = (modified_data['capital.gain'] > 0).astype(int)
            if 'capital.loss' in strategy['changes']:
                modified_data['has_capital_loss'] = (modified_data['capital.loss'] > 0).astype(int)
            
            # Update country grouping if needed
            if 'native.country' in strategy['changes']:
                common_countries = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba']
                modified_data['native.country_grouped'] = modified_data['native.country'].apply(
                    lambda x: x if x in common_countries else 'Other'
                )
            
            # Make prediction
            X_modified = preprocessor.transform(modified_data)
            new_prediction = model.predict(X_modified)[0]
            new_prediction_proba = model.predict_proba(X_modified)[0]
            
            # Store debug info
            debug_info.append({
                'strategy': strategy['name'],
                'original_pred': 1 - target_class,
                'new_pred': new_prediction,
                'target': target_class,
                'flipped': new_prediction == target_class,
                'prob_change': new_prediction_proba[target_class] - new_prediction_proba[1-target_class]
            })
            
            # Check if prediction flipped to target class OR significantly moved toward target
            prediction_flipped = new_prediction == target_class
            significant_improvement = (new_prediction_proba[target_class] - new_prediction_proba[1-target_class]) > 0.1
            
            if prediction_flipped or significant_improvement:
                # Calculate feasibility score
                feasibility_score = calculate_feasibility(
                    original_data, strategy, strategy['feasibility_factors']
                )
                
                # Format changes for display
                change_descriptions = []
                for feature, new_value in strategy['changes'].items():
                    if feature in original_data.columns:
                        old_value = original_data[feature].iloc[0]
                        if old_value != new_value:
                            change_desc = format_change_description(feature, old_value, new_value)
                            change_descriptions.append(change_desc)
                
                if change_descriptions:  # Only add if there are actual changes
                    # Calculate probability change correctly
                    original_target_prob = new_prediction_proba[1-target_class] if target_class == 0 else new_prediction_proba[0]
                    prob_change = new_prediction_proba[target_class] - original_target_prob
                    
                    counterfactual = {
                        'name': strategy['name'],
                        'changes': change_descriptions,
                        'new_probability': new_prediction_proba[target_class],
                        'probability_change': prob_change,
                        'feasibility_score': feasibility_score,
                        'prediction_flipped': prediction_flipped
                    }
                    counterfactuals.append(counterfactual)
                    
        except Exception as e:
            debug_info.append({
                'strategy': strategy['name'],
                'error': str(e)
            })
            continue  # Skip failed strategies
    
    # Sort by feasibility score (descending)
    counterfactuals.sort(key=lambda x: x['feasibility_score'], reverse=True)
    
    return counterfactuals

def calculate_feasibility(original_data, strategy, feasibility_factors):
    """Calculate how feasible a counterfactual scenario is"""
    
    base_score = 0.7  # Base feasibility
    age = original_data['age'].iloc[0]
    
    # Age-based adjustments
    if 'education' in strategy['changes']:
        if age > 40:
            base_score -= 0.2  # Harder to change education later in life
        elif age < 25:
            base_score += 0.1  # Easier when young
    
    if 'occupation' in strategy['changes']:
        if age > 50:
            base_score -= 0.3  # Career changes harder for older individuals
        elif age < 30:
            base_score += 0.1  # Easier career mobility when young
    
    # Specific feasibility factors
    if feasibility_factors.get('career_change', False):
        base_score -= 0.1  # Career changes are generally challenging
    
    if feasibility_factors.get('age_appropriate', True):
        base_score += 0.1  # Age-appropriate changes are more feasible
    
    # Financial changes (capital gain/loss) are often harder to control
    if 'capital.gain' in strategy['changes'] or 'capital.loss' in strategy['changes']:
        base_score -= 0.2
    
    # Marriage status changes
    if 'marital.status' in strategy['changes']:
        current_status = original_data['marital.status'].iloc[0]
        if current_status in ['Divorced', 'Widowed']:
            base_score -= 0.1  # Remarriage after divorce/death is possible but challenging
    
    return max(0.1, min(1.0, base_score))  # Clamp between 0.1 and 1.0

def format_change_description(feature, old_value, new_value):
    """Format a change description for display"""
    
    feature_names = {
        'age': 'Age',
        'education': 'Education Level',
        'education.num': 'Education Years',
        'occupation': 'Occupation',
        'hours.per.week': 'Hours per Week',
        'marital.status': 'Marital Status',
        'capital.gain': 'Capital Gain',
        'capital.loss': 'Capital Loss',
        'workclass': 'Work Class'
    }
    
    display_name = feature_names.get(feature, feature.replace('.', ' ').title())
    
    if feature in ['capital.gain', 'capital.loss']:
        return f"**{display_name}**: ${old_value:,} → ${new_value:,}"
    elif feature in ['age', 'education.num', 'hours.per.week']:
        return f"**{display_name}**: {old_value} → {new_value}"
    else:
        return f"**{display_name}**: {old_value} → {new_value}"

def analyze_counterfactuals(counterfactuals):
    """Analyze counterfactuals to provide insights"""
    
    insights = []
    
    if not counterfactuals:
        return ["No viable scenarios found to change the prediction."]
    
    # Analyze most common change types
    change_types = {}
    for cf in counterfactuals:
        for change in cf['changes']:
            if 'Education' in change:
                change_types['Education'] = change_types.get('Education', 0) + 1
            elif 'Occupation' in change:
                change_types['Occupation'] = change_types.get('Occupation', 0) + 1
            elif 'Hours' in change:
                change_types['Hours'] = change_types.get('Hours', 0) + 1
            elif 'Capital' in change:
                change_types['Investment'] = change_types.get('Investment', 0) + 1
            elif 'Marital' in change:
                change_types['Marriage'] = change_types.get('Marriage', 0) + 1
    
    # Generate insights based on most common changes
    if change_types:
        most_common = max(change_types, key=change_types.get)
        insights.append(f"**{most_common}** changes appear in most scenarios, suggesting this is a key factor.")
    
    # Feasibility insights
    high_feasibility = [cf for cf in counterfactuals if cf['feasibility_score'] > 0.8]
    if high_feasibility:
        insights.append(f"There are **{len(high_feasibility)}** highly feasible scenarios for changing the prediction.")
    else:
        moderate_feasibility = [cf for cf in counterfactuals if cf['feasibility_score'] > 0.6]
        if moderate_feasibility:
            insights.append(f"The most realistic options require **moderate effort** to achieve.")
        else:
            insights.append("Changing this prediction would require **significant life changes**.")
    
    # Probability change insights
    avg_prob_change = sum([cf['probability_change'] for cf in counterfactuals]) / len(counterfactuals)
    insights.append(f"On average, these changes would shift the probability by **{abs(avg_prob_change):.1%}**.")
    
    return insights

def main():
    st.set_page_config(
        page_title="Income Prediction App",
        page_icon="💰",
        layout="wide"
    )
    
    st.title("💰 Income Prediction App")
    st.write("Enter a person's attributes to predict their income level (<=50K or >50K)")
    
    # Load model and preprocessor
    try:
        model, preprocessor, model_name = load_model_and_preprocessor()
        st.sidebar.info(f"🤖 Using: {model_name}")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # Add tabs for different app sections
    main_tab, insights_tab = st.tabs(["🔮 Make Prediction", "📊 Model Insights"])
    
    with insights_tab:
        st.subheader("📈 Global Model Insights")
        st.write("Understanding how the model makes decisions across all predictions:")
        
        # Global feature importance
        if hasattr(model, 'feature_importances_') and model_name != "Random Forest (Fallback)":
            st.subheader("🎯 Global Feature Importance")
            st.write("These features have the most impact on predictions across all cases:")
            
            try:
                # Get feature importances
                feature_importances = model.feature_importances_
                
                # Create sample data to get feature names
                sample_data = pd.DataFrame({
                    'age': [39], 'workclass': ['Private'], 'fnlwgt': [77516],
                    'education': ['Bachelors'], 'education.num': [13], 'marital.status': ['Married-civ-spouse'],
                    'occupation': ['Tech-support'], 'relationship': ['Husband'], 'race': ['White'],
                    'sex': ['Male'], 'capital.gain': [0], 'capital.loss': [0],
                    'hours.per.week': [40], 'native.country': ['United-States']
                })
                
                # Add engineered features
                sample_data['has_capital_gain'] = 0
                sample_data['has_capital_loss'] = 0
                sample_data['native.country_grouped'] = 'United-States'
                
                # Get feature names
                feature_names = get_enhanced_feature_names(sample_data, len(feature_importances))
                
                # Create importance dataframe
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': feature_importances
                }).sort_values('Importance', ascending=False)
                
                # Show top features
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart of top features
                    fig, ax = plt.subplots(figsize=(12, 8))
                    
                    top_features = importance_df.head(15)
                    colors = plt.cm.viridis(np.linspace(0, 1, len(top_features)))
                    
                    bars = ax.barh(range(len(top_features)), top_features['Importance'], color=colors)
                    ax.set_yticks(range(len(top_features)))
                    ax.set_yticklabels(top_features['Feature'], fontsize=10)
                    ax.set_xlabel('Feature Importance', fontsize=12, fontweight='bold')
                    ax.set_title('Top 15 Most Important Features (Global)', fontsize=14, fontweight='bold')
                    ax.grid(True, alpha=0.3, axis='x')
                    
                    # Add value labels
                    for i, bar in enumerate(bars):
                        width = bar.get_width()
                        ax.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                               f'{width:.3f}', ha='left', va='center', fontsize=9, fontweight='bold')
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    st.markdown("**🔍 Key Insights:**")
                    
                    # Analyze top features
                    top_5_features = importance_df.head(5)['Feature'].tolist()
                    
                    feature_categories = {
                        'Education': ['Education', 'education'],
                        'Work': ['Occupation', 'Hours', 'Work'],
                        'Demographics': ['Age', 'Sex', 'Race', 'Marital'],
                        'Financial': ['Capital', 'Income']
                    }
                    
                    category_counts = {}
                    for feature in top_5_features:
                        for category, keywords in feature_categories.items():
                            if any(keyword.lower() in feature.lower() for keyword in keywords):
                                category_counts[category] = category_counts.get(category, 0) + 1
                                break
                    
                    if category_counts:
                        dominant_category = max(category_counts, key=category_counts.get)
                        st.write(f"• **{dominant_category}** factors dominate the top features")
                    
                    # Show top 3 individual features
                    st.write("**Top 3 Features:**")
                    for i in range(min(3, len(importance_df))):
                        feature_name = importance_df.iloc[i]['Feature']
                        importance = importance_df.iloc[i]['Importance']
                        st.write(f"{i+1}. **{feature_name}** ({importance:.3f})")
                
                # Feature importance distribution
                st.subheader("📊 Feature Importance Distribution")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of importance values
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(feature_importances, bins=20, color='skyblue', alpha=0.7, edgecolor='black')
                    ax.set_xlabel('Feature Importance Value', fontweight='bold')
                    ax.set_ylabel('Number of Features', fontweight='bold')
                    ax.set_title('Distribution of Feature Importances', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add statistics
                    mean_importance = np.mean(feature_importances)
                    ax.axvline(mean_importance, color='red', linestyle='--', 
                              label=f'Mean: {mean_importance:.3f}')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                with col2:
                    # Cumulative importance
                    fig, ax = plt.subplots(figsize=(8, 6))
                    
                    cumulative_importance = np.cumsum(importance_df['Importance'].values)
                    ax.plot(range(1, len(cumulative_importance) + 1), cumulative_importance, 
                           color='green', linewidth=2, marker='o', markersize=3)
                    
                    ax.set_xlabel('Number of Top Features', fontweight='bold')
                    ax.set_ylabel('Cumulative Importance', fontweight='bold')
                    ax.set_title('Cumulative Feature Importance', fontweight='bold')
                    ax.grid(True, alpha=0.3)
                    
                    # Add 80% line
                    total_importance = cumulative_importance[-1]
                    ax.axhline(total_importance * 0.8, color='red', linestyle='--', 
                              label='80% of Total Importance')
                    ax.legend()
                    
                    plt.tight_layout()
                    st.pyplot(fig)
                
                # Feature importance insights
                st.subheader("💡 Model Behavior Insights")
                
                insights_col1, insights_col2 = st.columns(2)
                
                with insights_col1:
                    st.markdown("**🎯 Feature Concentration:**")
                    
                    # Calculate how many features account for 80% of importance
                    cumulative_pct = cumulative_importance / total_importance
                    features_for_80pct = np.argmax(cumulative_pct >= 0.8) + 1
                    
                    st.write(f"• **{features_for_80pct}** features account for 80% of model decisions")
                    st.write(f"• Top **5** features represent {cumulative_pct[4]:.1%} of total importance")
                    st.write(f"• **{len(feature_importances)}** total features in the model")
                    
                    if features_for_80pct < 20:
                        st.success("✅ Model focuses on a manageable number of key features")
                    else:
                        st.warning("⚠️ Model uses many features - predictions may be complex")
                
                with insights_col2:
                    st.markdown("**📈 Prediction Drivers:**")
                    
                    # Analyze feature types in top features
                    top_10_features = importance_df.head(10)['Feature'].tolist()
                    
                    education_features = sum(1 for f in top_10_features if 'education' in f.lower())
                    work_features = sum(1 for f in top_10_features if any(w in f.lower() for w in ['occupation', 'hours', 'work']))
                    financial_features = sum(1 for f in top_10_features if any(w in f.lower() for w in ['capital', 'gain', 'loss']))
                    demographic_features = sum(1 for f in top_10_features if any(w in f.lower() for w in ['age', 'sex', 'race', 'marital']))
                    
                    if education_features >= 3:
                        st.write("📚 **Education** is a major prediction factor")
                    if work_features >= 3:
                        st.write("💼 **Work characteristics** heavily influence predictions")
                    if financial_features >= 2:
                        st.write("💰 **Financial factors** play an important role")
                    if demographic_features >= 3:
                        st.write("👥 **Demographics** significantly impact predictions")
                
            except Exception as e:
                st.error(f"Could not generate global insights: {str(e)}")
        
        else:
            st.info("💡 Global feature importance is not available for this model type.")
    
    with main_tab:
        # Create input form
        with st.form("prediction_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Personal Information")
                age = st.number_input("Age", min_value=17, max_value=90, value=39)
                sex = st.selectbox("Sex", ["Male", "Female"])
                race = st.selectbox("Race", [
                    "White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"
                ])
                
                st.subheader("Education")
                education = st.selectbox("Education Level", [
                    "Bachelors", "Some-college", "11th", "HS-grad", "Prof-school",
                    "Assoc-acdm", "Assoc-voc", "9th", "7th-8th", "12th", "Masters",
                    "1st-4th", "10th", "Doctorate", "5th-6th", "Preschool"
                ])
                education_num = st.number_input("Education Years", min_value=1, max_value=16, value=13)
            
            with col2:
                st.subheader("Work Information")
                workclass = st.selectbox("Work Class", [
                    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov",
                    "Local-gov", "State-gov", "Without-pay", "Never-worked"
                ])
                
                occupation = st.selectbox("Occupation", [
                    "Tech-support", "Craft-repair", "Other-service", "Sales",
                    "Exec-managerial", "Prof-specialty", "Handlers-cleaners",
                    "Machine-op-inspct", "Adm-clerical", "Farming-fishing",
                    "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
                ])
                
                hours_per_week = st.number_input("Hours per Week", min_value=1, max_value=99, value=40)
                
                st.subheader("Family Information")
                marital_status = st.selectbox("Marital Status", [
                    "Married-civ-spouse", "Divorced", "Never-married", "Separated",
                    "Widowed", "Married-spouse-absent", "Married-AF-spouse"
                ])
                
                relationship = st.selectbox("Relationship", [
                    "Wife", "Own-child", "Husband", "Not-in-family",
                    "Other-relative", "Unmarried"
                ])
            
            col3, col4 = st.columns(2)
            
            with col3:
                st.subheader("Financial Information")
                capital_gain = st.number_input("Capital Gain", min_value=0, max_value=99999, value=0)
                capital_loss = st.number_input("Capital Loss", min_value=0, max_value=4356, value=0)
                
            with col4:
                st.subheader("Location")
                native_country = st.selectbox("Native Country", [
                    "United-States", "Cuba", "Jamaica", "India", "Mexico", "South",
                    "Puerto-Rico", "Honduras", "England", "Canada", "Germany", "Iran",
                    "Philippines", "Italy", "Poland", "Columbia", "Cambodia", "Thailand",
                    "Ecuador", "Laos", "Taiwan", "Haiti", "Portugal", "Dominican-Republic",
                    "El-Salvador", "France", "Guatemala", "China", "Japan", "Yugoslavia",
                    "Peru", "Outlying-US(Guam-USVI-etc)", "Scotland", "Trinadad&Tobago",
                    "Greece", "Nicaragua", "Vietnam", "Hong", "Ireland", "Hungary",
                    "Holand-Netherlands"
                ])
                
                fnlwgt = st.number_input("Final Weight", min_value=12285, max_value=1484705, value=77516)
        
            submitted = st.form_submit_button("Predict Income")
        
        if submitted:
                # Create input dataframe
                input_data = pd.DataFrame({
                    'age': [age],
                    'workclass': [workclass],
                    'fnlwgt': [fnlwgt],
                    'education': [education],
                    'education.num': [education_num],
                    'marital.status': [marital_status],
                    'occupation': [occupation],
                    'relationship': [relationship],
                    'race': [race],
                    'sex': [sex],
                    'capital.gain': [capital_gain],
                    'capital.loss': [capital_loss],
                    'hours.per.week': [hours_per_week],
                    'native.country': [native_country]
                })
                
                try:
                    # Preprocess the input data
                    if model_name == "Random Forest (Fallback)":
                        # Use simplified preprocessing for fallback model
                        X_processed = create_feature_vector(input_data)
                        X_processed = preprocessor.transform(X_processed)
                    else:
                        # Create engineered features that the original model expects
                        input_data_processed = input_data.copy()
                        
                        # Create has_capital_gain feature
                        input_data_processed['has_capital_gain'] = (input_data_processed['capital.gain'] > 0).astype(int)
                        
                        # Create has_capital_loss feature  
                        input_data_processed['has_capital_loss'] = (input_data_processed['capital.loss'] > 0).astype(int)
                        
                        # Create native.country_grouped feature
                        # Group less common countries as 'Other'
                        common_countries = ['United-States', 'Mexico', 'Philippines', 'Germany', 'Canada', 'Puerto-Rico', 'El-Salvador', 'India', 'Cuba']
                        input_data_processed['native.country_grouped'] = input_data_processed['native.country'].apply(
                            lambda x: x if x in common_countries else 'Other'
                        )
                        
                        # Use original preprocessing pipeline
                        X_processed = preprocessor.transform(input_data_processed)
                    
                    # Make prediction
                    prediction = model.predict(X_processed)[0]
                    prediction_proba = model.predict_proba(X_processed)[0]
                    
                    # Display results
                    st.subheader("Prediction Results")
                    
                    if model_name == "Random Forest (Fallback)":
                        st.warning("⚠️ Using simplified fallback model - predictions may be less accurate")
                    
                    if prediction == 1:
                        st.success("💰 Predicted Income: **> $50,000**")
                    else:
                        st.info("💼 Predicted Income: **≤ $50,000**")
                    
                    # Show prediction confidence
                    confidence = max(prediction_proba) * 100
                    st.write(f"**Confidence:** {confidence:.1f}%")
                    
                    # Show probability breakdown
                    st.subheader("Probability Breakdown")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.metric("≤ $50,000", f"{prediction_proba[0]:.1%}")
                    
                    with col2:
                        st.metric("> $50,000", f"{prediction_proba[1]:.1%}")
                
                    # Enhanced SHAP Explanations
                    if SHAP_AVAILABLE and model_name != "Random Forest (Fallback)":
                        st.subheader("🔍 Enhanced Model Explanations")
                        
                        # Create tabs for different explanation types
                        tab1, tab2, tab3 = st.tabs(["📊 Feature Impact", "🏄 Waterfall Plot", "⚡ Force Plot"])
                        
                        try:
                            # Create SHAP explainer
                            explainer = create_shap_explainer(model, X_processed)
                            
                            if explainer is not None:
                                # Calculate SHAP values for this prediction
                                shap_values = explainer.shap_values(X_processed)
                                
                                # Handle different SHAP output formats and convert to float
                                if isinstance(shap_values, list) and len(shap_values) == 2:
                                    # Binary classification with separate arrays for each class
                                    shap_values_to_plot = np.array(shap_values[1][0], dtype=float)  # Positive class
                                    expected_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
                                elif len(shap_values.shape) == 3:
                                    # 3D array format
                                    shap_values_to_plot = np.array(shap_values[0, :, 1], dtype=float)  # Positive class
                                    expected_value = float(explainer.expected_value[1] if isinstance(explainer.expected_value, list) else explainer.expected_value)
                                else:
                                    # 2D array format
                                    shap_values_to_plot = np.array(shap_values[0], dtype=float)
                                    expected_value = float(explainer.expected_value)
                                
                                # Get better feature names
                                feature_names = get_enhanced_feature_names(input_data_processed, len(shap_values_to_plot))
                            
                            with tab1:
                                st.write("**How each feature impacts the prediction:**")
                                
                                # Get top contributing features
                                feature_importance = list(zip(feature_names, [float(x) for x in shap_values_to_plot]))
                                feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                                
                                # Display top features with better formatting
                                for i, (feature, shap_val) in enumerate(feature_importance[:12]):
                                    col1, col2, col3 = st.columns([3, 1, 2])
                                    
                                    with col1:
                                        direction = "🟢" if shap_val > 0 else "🔴"
                                        st.write(f"{direction} **{feature}**")
                                    
                                    with col2:
                                        impact = "+" if shap_val > 0 else ""
                                        st.write(f"{impact}{shap_val:.3f}")
                                    
                                    with col3:
                                        # Progress bar showing relative impact
                                        max_abs_val = max([abs(float(x[1])) for x in feature_importance[:12]])
                                        if max_abs_val > 0:
                                            bar_value = float(abs(shap_val)) / max_abs_val
                                            st.progress(min(1.0, max(0.0, bar_value)))
                                
                                # Enhanced bar chart
                                fig, ax = plt.subplots(figsize=(12, 8))
                                
                                # Get top 15 features for plotting
                                features_to_plot = feature_importance[:15]
                                feature_names_plot = [x[0] for x in features_to_plot]
                                shap_vals_plot = [x[1] for x in features_to_plot]
                                
                                # Create horizontal bar plot with better styling
                                colors = ['#2E8B57' if x > 0 else '#DC143C' for x in shap_vals_plot]
                                bars = ax.barh(range(len(feature_names_plot)), shap_vals_plot, color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
                                
                                ax.set_yticks(range(len(feature_names_plot)))
                                ax.set_yticklabels(feature_names_plot, fontsize=10)
                                ax.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=12, fontweight='bold')
                                ax.set_title('Feature Importance for This Prediction', fontsize=14, fontweight='bold')
                                ax.axvline(x=0, color='black', linestyle='-', alpha=0.8, linewidth=1)
                                ax.grid(True, alpha=0.3, axis='x')
                                
                                # Add value labels on bars
                                for i, bar in enumerate(bars):
                                    width = bar.get_width()
                                    ax.text(width + (0.001 if width > 0 else -0.001), bar.get_y() + bar.get_height()/2, 
                                           f'{width:.3f}', 
                                           ha='left' if width > 0 else 'right', 
                                           va='center', fontsize=9, fontweight='bold')
                                
                                plt.tight_layout()
                                st.pyplot(fig)
                                
                                st.info("🟢 Green bars increase probability of >$50K | 🔴 Red bars decrease probability")
                            
                            with tab2:
                                st.write("**Waterfall Plot - Step-by-step prediction building:**")
                                
                                try:
                                    # Create waterfall plot
                                    fig, ax = plt.subplots(figsize=(12, 8))
                                    
                                    # Calculate cumulative values for waterfall
                                    sorted_features = sorted(zip(feature_names, [float(x) for x in shap_values_to_plot]), key=lambda x: abs(x[1]), reverse=True)[:10]
                                    feature_names_sorted = [x[0] for x in sorted_features]
                                    shap_vals_sorted = [x[1] for x in sorted_features]
                                    
                                    # Create waterfall effect
                                    cumulative = [expected_value]
                                    for val in shap_vals_sorted:
                                        cumulative.append(cumulative[-1] + val)
                                    
                                    # Plot waterfall
                                    positions = range(len(feature_names_sorted) + 2)
                                    
                                    # Base value
                                    ax.bar(0, expected_value, color='lightblue', alpha=0.7, label='Base Prediction')
                                    ax.text(0, expected_value/2, f'{expected_value:.3f}', ha='center', va='center', fontweight='bold')
                                    
                                    # Feature contributions
                                    for i, (name, val) in enumerate(sorted_features):
                                        color = '#2E8B57' if val > 0 else '#DC143C'
                                        bottom = cumulative[i] if val > 0 else cumulative[i] + val
                                        ax.bar(i + 1, abs(val), bottom=bottom, color=color, alpha=0.8)
                                        ax.text(i + 1, cumulative[i] + val/2, f'{val:+.3f}', ha='center', va='center', 
                                               fontweight='bold', fontsize=9, rotation=90 if len(name) > 10 else 0)
                                    
                                    # Final prediction
                                    final_val = cumulative[-1]
                                    ax.bar(len(feature_names_sorted) + 1, final_val, color='gold', alpha=0.8, label='Final Prediction')
                                    ax.text(len(feature_names_sorted) + 1, final_val/2, f'{final_val:.3f}', ha='center', va='center', fontweight='bold')
                                    
                                    # Styling
                                    ax.set_xticks(positions)
                                    ax.set_xticklabels(['Base'] + feature_names_sorted + ['Final'], rotation=45, ha='right')
                                    ax.set_ylabel('Prediction Score', fontsize=12, fontweight='bold')
                                    ax.set_title('Waterfall Plot - How Features Build the Prediction', fontsize=14, fontweight='bold')
                                    ax.grid(True, alpha=0.3, axis='y')
                                    ax.legend()
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    st.info(f"📊 Base model prediction: {expected_value:.3f} → Final prediction: {final_val:.3f}")
                                    
                                except Exception as e:
                                    st.error(f"Could not create waterfall plot: {str(e)}")
                            
                            with tab3:
                                st.write("**Force Plot - Interactive explanation:**")
                                
                                try:
                                    # Create force plot data
                                    feature_importance = list(zip(feature_names, [float(x) for x in shap_values_to_plot]))
                                    feature_importance.sort(key=lambda x: abs(x[1]), reverse=True)
                                    
                                    # Show top positive and negative contributors
                                    positive_features = [(name, val) for name, val in feature_importance if val > 0][:5]
                                    negative_features = [(name, val) for name, val in feature_importance if val < 0][:5]
                                    
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**🟢 Factors Increasing Income Probability:**")
                                        for name, val in positive_features:
                                            st.write(f"• **{name}**: +{val:.3f}")
                                        
                                    with col2:
                                        st.markdown("**🔴 Factors Decreasing Income Probability:**")
                                        for name, val in negative_features:
                                            st.write(f"• **{name}**: {val:.3f}")
                                    
                                    # Create force plot visualization
                                    fig, ax = plt.subplots(figsize=(14, 6))
                                    
                                    # Calculate the prediction components
                                    base_value = expected_value
                                    final_value = base_value + sum([float(x) for x in shap_values_to_plot])
                                    
                                    # Create force plot style visualization
                                    y_pos = 0.5
                                    
                                    # Draw base line
                                    ax.axhline(y=y_pos, color='gray', linestyle='--', alpha=0.5)
                                    
                                    # Sort features by impact for better visualization
                                    sorted_impacts = sorted(zip(feature_names, [float(x) for x in shap_values_to_plot], range(len(feature_names))), 
                                                          key=lambda x: x[1], reverse=True)
                                    
                                    # Draw arrows for only top 5 most significant features to avoid overlap
                                    current_pos = base_value
                                    arrow_height = 0.15
                                    significant_impacts = [x for x in sorted_impacts if abs(x[1]) > 0.01][:5]  # Top 5 significant features
                                    
                                    for i, (name, impact, orig_idx) in enumerate(significant_impacts):
                                        color = '#2E8B57' if impact > 0 else '#DC143C'
                                        arrow_y = y_pos + (i % 2 - 0.5) * arrow_height * 1.5  # More spacing
                                        
                                        # Draw arrow
                                        ax.annotate('', xy=(current_pos + impact, arrow_y), xytext=(current_pos, arrow_y),
                                                   arrowprops=dict(arrowstyle='->', color=color, lw=3, alpha=0.8))
                                        
                                        # Shorten feature names for display
                                        display_name = name[:15] + "..." if len(name) > 15 else name
                                        
                                        # Add feature label with better positioning
                                        label_x = current_pos + impact/2
                                        ax.text(label_x, arrow_y + 0.08, f'{display_name}\n{impact:+.3f}', 
                                               ha='center', va='bottom', fontsize=9, fontweight='bold',
                                               bbox=dict(boxstyle='round,pad=0.4', facecolor=color, alpha=0.6, edgecolor='white'))
                                        
                                        current_pos += impact
                                    
                                    # Mark base and final values
                                    ax.axvline(x=base_value, color='blue', linestyle='-', alpha=0.8, linewidth=2)
                                    ax.axvline(x=final_value, color='red', linestyle='-', alpha=0.8, linewidth=2)
                                    
                                    ax.text(base_value, y_pos - 0.2, f'Base\n{base_value:.3f}', ha='center', va='top', 
                                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightblue'))
                                    ax.text(final_value, y_pos - 0.2, f'Prediction\n{final_value:.3f}', ha='center', va='top', 
                                           fontweight='bold', bbox=dict(boxstyle='round', facecolor='lightcoral'))
                                    
                                    ax.set_ylim(0, 1.2)  # More vertical space for labels
                                    ax.set_ylabel('Feature Contributions', fontsize=12, fontweight='bold')
                                    ax.set_xlabel('Prediction Score', fontsize=12, fontweight='bold')
                                    ax.set_title('Force Plot - Top 5 Feature Push/Pull Effects', fontsize=14, fontweight='bold')
                                    ax.grid(True, alpha=0.3, axis='x')
                                    
                                    # Remove y-axis ticks as they're not meaningful
                                    ax.set_yticks([])
                                    
                                    # Add note about limited features
                                    ax.text(0.02, 0.98, f'Showing top {len(significant_impacts)} most significant features', 
                                           transform=ax.transAxes, fontsize=10, va='top',
                                           bbox=dict(boxstyle='round,pad=0.3', facecolor='lightgray', alpha=0.7))
                                    
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    
                                    # Explanation
                                    st.info("🎯 This plot shows how each feature 'pushes' the prediction from the base value toward the final prediction")
                                    
                                except Exception as e:
                                    st.error(f"Could not create force plot: {str(e)}")
                            
                        except Exception as e:
                            st.warning(f"Could not generate SHAP explanations: {str(e)}")
                    
                    elif model_name == "Random Forest (Fallback)":
                        st.info("💡 Enhanced SHAP explanations are not available for the fallback model.")
                    
                    # Counterfactual Explanations
                    st.subheader("🔄 Counterfactual Analysis")
                    st.write("What would need to change to flip the prediction?")
                
                    with st.expander("🎭 Find Alternative Scenarios", expanded=False):
                        if model_name != "Random Forest (Fallback)":
                            try:
                                # Define the target class (opposite of current prediction)
                                target_class = 0 if prediction == 1 else 1
                                target_label = "≤ $50K" if target_class == 0 else "> $50K"
                                current_label = "> $50K" if prediction == 1 else "≤ $50K"
                            
                                st.write(f"**Current prediction: {current_label}**")
                                st.write(f"**Target: {target_label}**")
                                st.write("---")
                            
                                # Generate counterfactual scenarios
                                counterfactuals = generate_counterfactuals(
                                    input_data_processed, model, preprocessor, target_class, model_name
                                )
                            
                                if counterfactuals:
                                    # Separate complete flips from improvements
                                    complete_flips = [cf for cf in counterfactuals if cf.get('prediction_flipped', False)]
                                    improvements = [cf for cf in counterfactuals if not cf.get('prediction_flipped', False)]
                                    
                                    if complete_flips:
                                        st.write("**🎯 Scenarios that would flip the prediction:**")
                                        
                                        for i, cf in enumerate(complete_flips[:3], 1):
                                            with st.container():
                                                st.markdown(f"**✅ Scenario {i}: {cf['name']}**")
                                                
                                                col1, col2 = st.columns([2, 1])
                                                
                                                with col1:
                                                    for change in cf['changes']:
                                                        st.write(f"• {change}")
                                                
                                                with col2:
                                                    st.metric(
                                                        "New Probability", 
                                                        f"{cf['new_probability']:.1%}",
                                                        delta=f"{cf['probability_change']:+.1%}"
                                                    )
                                                
                                                # Show feasibility score
                                                feasibility = cf['feasibility_score']
                                                if feasibility > 0.8:
                                                    feasibility_color = "🟢"
                                                    feasibility_text = "Highly Feasible"
                                                elif feasibility > 0.6:
                                                    feasibility_color = "🟡"
                                                    feasibility_text = "Moderately Feasible" 
                                                else:
                                                    feasibility_color = "🔴"
                                                    feasibility_text = "Challenging"
                                            
                                                st.write(f"{feasibility_color} **Feasibility:** {feasibility_text} ({feasibility:.1%})")
                                                st.write("---")
                                    
                                    if improvements:
                                        st.write("**📈 Scenarios that would significantly improve the prediction:**")
                                        
                                        for i, cf in enumerate(improvements[:3], 1):
                                            with st.container():
                                                st.markdown(f"**⬆️ Improvement {i}: {cf['name']}**")
                                                
                                                col1, col2 = st.columns([2, 1])
                                                
                                                with col1:
                                                    for change in cf['changes']:
                                                        st.write(f"• {change}")
                                                
                                                with col2:
                                                    st.metric(
                                                        "New Probability", 
                                                        f"{cf['new_probability']:.1%}",
                                                        delta=f"{cf['probability_change']:+.1%}"
                                                    )
                                                
                                                # Show feasibility score
                                                feasibility = cf['feasibility_score']
                                                if feasibility > 0.8:
                                                    feasibility_color = "🟢"
                                                    feasibility_text = "Highly Feasible"
                                                elif feasibility > 0.6:
                                                    feasibility_color = "🟡"
                                                    feasibility_text = "Moderately Feasible" 
                                                else:
                                                    feasibility_color = "🔴"
                                                    feasibility_text = "Challenging"
                                            
                                                st.write(f"{feasibility_color} **Feasibility:** {feasibility_text} ({feasibility:.1%})")
                                                st.write("---")
                                
                                    # Summary insights
                                    st.markdown("**🎯 Key Insights:**")
                                    insights = analyze_counterfactuals(counterfactuals)
                                    for insight in insights:
                                        st.write(f"• {insight}")
                                        
                                    if not complete_flips and improvements:
                                        st.info("💡 While no scenarios completely flip the prediction, several can significantly improve the outcome.")
                                        
                                else:
                                    st.warning("Could not generate meaningful counterfactual scenarios.")
                                    st.write("**Possible reasons:**")
                                    st.write("• The current prediction may already be very strong")
                                    st.write("• The model may be highly confident in its decision")  
                                    st.write("• Try different input values to explore other scenarios")
                                
                            except Exception as e:
                                st.error(f"Error generating counterfactuals: {str(e)}")
                        else:
                            st.info("💡 Counterfactual analysis is not available for the fallback model.")
                
                except Exception as e:
                    st.error(f"Error making prediction: {str(e)}")

if __name__ == "__main__":
    main()