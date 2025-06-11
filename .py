#!/usr/bin/env python3
"""
Jupiter ML Internship Challenge - Complete Solution
Generates 25,000 synthetic credit records and trains ML models
"""
 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')
 
def create_jupiter_dataset():
    """Generate synthetic credit dataset with 25,000 records"""
    np.random.seed(42)
    n_samples = 25000
    
    # Demographics
    customers = {
        'customer_id': [f"CUST_{i:06d}" for i in range(1, n_samples + 1)],
        'age': np.random.normal(35, 12, n_samples).clip(18, 75).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_samples, p=[0.6, 0.4]),
        'location': np.random.choice(['Tier1', 'Tier2', 'Tier3', 'Metro'], n_samples, p=[0.25, 0.35, 0.25, 0.15])
    }
    
    # Financial features with realistic distributions
    income = np.random.lognormal(np.log(40000), 0.7, n_samples).clip(15000, 500000).astype(int)
    customers['monthly_income'] = income
    
    # EMI outflow (15-45% of income)
    emi_ratio = np.random.uniform(0.15, 0.45, n_samples)
    customers['monthly_emi_outflow'] = (income * emi_ratio).astype(int)
    
    # Credit behavior features
    customers['current_outstanding'] = np.random.lognormal(11, 1, n_samples).clip(10000, 1500000).astype(int)
    customers['credit_utilization_ratio'] = np.random.beta(3, 4, n_samples).clip(0, 1)
    customers['num_open_loans'] = np.random.poisson(2.5, n_samples).clip(0, 12)
    customers['repayment_history_score'] = np.random.normal(72, 18, n_samples).clip(20, 100).astype(int)
    customers['dpd_last_3_months'] = np.random.exponential(8, n_samples).clip(0, 90).astype(int)
    customers['num_hard_inquiries_last_6m'] = np.random.poisson(2, n_samples).clip(0, 8)
    customers['recent_credit_card_usage'] = np.random.lognormal(9.5, 1, n_samples).clip(1000, 150000).astype(int)
    customers['recent_loan_disbursed_amount'] = np.random.lognormal(10.5, 1.5, n_samples).clip(0, 800000).astype(int)
    customers['total_credit_limit'] = np.random.lognormal(11.2, 0.9, n_samples).clip(25000, 750000).astype(int)
    customers['months_since_last_default'] = np.random.exponential(25, n_samples).clip(0, 120).astype(int)
    
    df = pd.DataFrame(customers)
    
    # Create target variable using credit risk heuristics
    # Payment behavior (40% weight) - DPD and repayment score
    payment_risk = (df['dpd_last_3_months'] / 90) * 0.6 + (1 - df['repayment_history_score'] / 100) * 0.4
    
    # Credit utilization (25% weight)
    util_risk = df['credit_utilization_ratio']
    
    # Credit inquiries (15% weight)
    inquiry_risk = np.minimum(1, df['num_hard_inquiries_last_6m'] / 6)
    
    # Debt burden (10% weight)
    debt_burden = df['monthly_emi_outflow'] / df['monthly_income']
    
    # Default history (10% weight)
    default_risk = 1 - np.minimum(1, df['months_since_last_default'] / 48)
    
    # Composite risk score
    total_risk = (payment_risk * 0.40 + util_risk * 0.25 + 
                  inquiry_risk * 0.15 + debt_burden * 0.10 + default_risk * 0.10)
    
    # Add noise for realism
    noise = np.random.normal(0, 0.05, len(df))
    risk_score = (total_risk + noise).clip(0, 1)
    
    # Convert to target classes
    def risk_to_movement(risk):
        if risk <= 0.35:
            return 'increase'
        elif risk >= 0.65:
            return 'decrease'
        else:
            return 'stable'
    
    df['target_credit_score_movement'] = [risk_to_movement(r) for r in risk_score]
    return df
 
def train_models(df):
    """Train and evaluate machine learning models"""
    
    # Prepare features
    feature_cols = [col for col in df.columns if col not in ['customer_id', 'target_credit_score_movement']]
    X = df[feature_cols].copy()
    y = df['target_credit_score_movement']
    
    # Encode categorical variables
    le_gender = LabelEncoder()
    le_location = LabelEncoder()
    X['gender'] = le_gender.fit_transform(X['gender'])
    X['location'] = le_location.fit_transform(X['location'])
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train models
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=50, max_depth=12, random_state=42, n_jobs=-1, class_weight='balanced'),
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=500, class_weight='balanced')
    }
    
    results = {}
    for name, model in models.items():
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        
        results[name] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_macro': f1_score(y_test, y_pred, average='macro'),
            'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': getattr(model, 'feature_importances_', None)
        }
    
    best_model = max(results.keys(), key=lambda k: results[k]['f1_macro'])
    return results, best_model, feature_cols
 
def main():
    print("JUPITER ML INTERNSHIP CHALLENGE")
    print("Credit Score Movement Prediction")
    print("=" * 50)
    
    # Generate dataset
    print("Generating 25,000 synthetic credit records...")
    df = create_jupiter_dataset()
    df.to_csv('jupiter_credit_dataset.csv', index=False)
    
    print(f"Dataset created: {len(df):,} records, {len(df.columns)-2} features")
    print("\nTarget distribution:")
    for target, count in df['target_credit_score_movement'].value_counts().items():
        print(f"  {target}: {count:,} ({count/len(df)*100:.1f}%)")
    
    # Train models
    print("\nTraining machine learning models...")
    results, best_model, feature_cols = train_models(df)
    
    # Results
    print(f"\nMODEL PERFORMANCE")
    print("-" * 30)
    for model_name, metrics in results.items():
        print(f"{model_name}:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  F1-Score (Macro): {metrics['f1_macro']:.3f}")
        print(f"  F1-Score (Weighted): {metrics['f1_weighted']:.3f}\n")
    
    print(f"BEST MODEL: {best_model}")
    print("=" * 30)
    best_results = results[best_model]
    print(f"Accuracy: {best_results['accuracy']:.3f}")
    print(f"F1-Score (Macro): {best_results['f1_macro']:.3f}")
    print(f"F1-Score (Weighted): {best_results['f1_weighted']:.3f}")
    
    print("\nClassification Report:")
    print(best_results['classification_report'])
    
    # Feature importance
    if best_results['feature_importance'] is not None:
        print("Top 8 Most Important Features:")
        importance_data = list(zip(feature_cols, best_results['feature_importance']))
        importance_data.sort(key=lambda x: x[1], reverse=True)
        
        for i, (feature, importance) in enumerate(importance_data[:8], 1):
            print(f"  {i}. {feature}: {importance:.4f}")
    
    # Business insights
    high_risk = df[df['target_credit_score_movement'] == 'decrease']
    growth_potential = df[df['target_credit_score_movement'] == 'increase']
    
    print(f"\nBUSINESS INSIGHTS")
    print("-" * 30)
    print(f"High Risk Customers: {len(high_risk):,}")
    print(f"  Avg Credit Utilization: {high_risk['credit_utilization_ratio'].mean():.1%}")
    print(f"  Avg DPD: {high_risk['dpd_last_3_months'].mean():.1f} days")
    
    print(f"\nGrowth Potential: {len(growth_potential):,}")
    print(f"  Avg Repayment Score: {growth_potential['repayment_history_score'].mean():.0f}")
    print(f"  Avg Income: ₹{growth_potential['monthly_income'].mean():,.0f}")
    
    print(f"\nRECOMMENDations:")
    print("1. Monitor customers with >70% credit utilization")
    print("2. Implement early warning for increasing DPD")
    print("3. Offer credit limit increases to growth segment")
    print("4. Create financial education for high-risk customers")
    
    print(f"\nCHALLENGE COMPLETED SUCCESSFULLY!")
    print(f"✓ Dataset: {len(df):,} records")
    print(f"✓ Features: {len(feature_cols)} variables")
    print(f"✓ Model: {best_results['f1_macro']:.1%} balanced F1-score")
    print(f"✓ Business insights and recommendations provided")
 
if __name__ == "__main__":
    main()
