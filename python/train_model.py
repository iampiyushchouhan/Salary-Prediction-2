#!/usr/bin/env python3
"""
Employee Salary Prediction Model Training Script
This script trains multiple ML models on salary data and saves the best performing model
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

class SalaryPredictor:
    def __init__(self):
        self.model = None
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def load_and_preprocess_data(self, filepath):
        """Load and preprocess the salary dataset"""
        print("Loading and preprocessing data...")
        
        # Load data
        df = pd.read_csv(filepath)
        print(f"Dataset shape: {df.shape}")
        print(f"Dataset columns: {df.columns.tolist()}")
        
        # Display basic statistics
        print("\nDataset Info:")
        print(df.info())
        print("\nSalary Statistics:")
        print(df['salary_inr'].describe())
        
        # Handle missing values
        df = df.dropna()
        
        # Feature engineering
        df['experience_category'] = pd.cut(df['experience_years'], 
                                         bins=[0, 2, 5, 10, 20], 
                                         labels=['Junior', 'Mid', 'Senior', 'Lead'])
        
        df['salary_category'] = pd.cut(df['salary_inr'], 
                                     bins=[0, 500000, 1000000, 1500000, float('inf')], 
                                     labels=['Low', 'Medium', 'High', 'Very High'])
        
        # Separate features and target
        feature_cols = ['job_title', 'experience_years', 'education', 'skills_level', 
                       'company_type', 'location', 'industry', 'work_mode']
        
        X = df[feature_cols].copy()
        y = df['salary_inr'].copy()
        
        # Encode categorical variables
        categorical_cols = ['job_title', 'education', 'skills_level', 'company_type', 
                           'location', 'industry', 'work_mode']
        
        for col in categorical_cols:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.encoders[col] = le
        
        # Scale numerical features
        numerical_cols = ['experience_years']
        scaler = StandardScaler()
        X[numerical_cols] = scaler.fit_transform(X[numerical_cols])
        self.scalers['numerical'] = scaler
        
        self.feature_names = X.columns.tolist()
        
        return X, y, df
    
    def train_models(self, X, y):
        """Train multiple models and select the best one"""
        print("\nTraining multiple models...")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Define models
        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(),
            'SVR': SVR()
        }
        
        # Train and evaluate models
        results = {}
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
            
            # Train on full training set
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Calculate metrics
            r2 = r2_score(y_test, y_pred)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            
            results[name] = {
                'model': model,
                'cv_score': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'r2_score': r2,
                'mse': mse,
                'mae': mae,
                'rmse': np.sqrt(mse)
            }
            
            print(f"  CV R² Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"  Test R² Score: {r2:.4f}")
            print(f"  RMSE: ₹{np.sqrt(mse):,.0f}")
        
        # Select best model based on R² score
        best_model_name = max(results.keys(), key=lambda x: results[x]['r2_score'])
        self.model = results[best_model_name]['model']
        
        print(f"\nBest Model: {best_model_name}")
        print(f"R² Score: {results[best_model_name]['r2_score']:.4f}")
        print(f"RMSE: ₹{results[best_model_name]['rmse']:,.0f}")
        
        return results, X_test, y_test
    
    def hyperparameter_tuning(self, X, y):
        """Perform hyperparameter tuning for the best model"""
        print("\nPerforming hyperparameter tuning...")
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Random Forest hyperparameter tuning
        rf_params = {
            'n_estimators': [100, 200, 300],
            'max_depth': [10, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        rf = RandomForestRegressor(random_state=42)
        rf_grid = GridSearchCV(rf, rf_params, cv=3, scoring='r2', n_jobs=-1, verbose=1)
        rf_grid.fit(X_train, y_train)
        
        self.model = rf_grid.best_estimator_
        
        print(f"Best parameters: {rf_grid.best_params_}")
        print(f"Best CV score: {rf_grid.best_score_:.4f}")
        
        # Test performance
        y_pred = self.model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        
        print(f"Final Test R² Score: {r2:.4f}")
        print(f"Final RMSE: ₹{rmse:,.0f}")
        
        return self.model
    
    def plot_results(self, results, X_test, y_test):
        """Plot model comparison and feature importance"""
        # Model comparison plot
        plt.figure(figsize=(15, 10))
        
        # Subplot 1: Model R² Comparison
        plt.subplot(2, 3, 1)
        model_names = list(results.keys())
        r2_scores = [results[name]['r2_score'] for name in model_names]
        
        bars = plt.bar(model_names, r2_scores, color='skyblue', alpha=0.7)
        plt.title('Model R² Score Comparison')
        plt.ylabel('R² Score')
        plt.xticks(rotation=45)
        plt.ylim(0, 1)
        
        # Add value labels on bars
        for bar, score in zip(bars, r2_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                    f'{score:.3f}', ha='center', va='bottom')
        
        # Subplot 2: RMSE Comparison
        plt.subplot(2, 3, 2)
        rmse_scores = [results[name]['rmse'] for name in model_names]
        bars = plt.bar(model_names, rmse_scores, color='lightcoral', alpha=0.7)
        plt.title('Model RMSE Comparison')
        plt.ylabel('RMSE (₹)')
        plt.xticks(rotation=45)
        
        # Add value labels on bars
        for bar, score in zip(bars, rmse_scores):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10000, 
                    f'₹{score:,.0f}', ha='center', va='bottom', rotation=90)
        
        # Subplot 3: Actual vs Predicted
        plt.subplot(2, 3, 3)
        y_pred = self.model.predict(X_test)
        plt.scatter(y_test, y_pred, alpha=0.6)
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Salary (₹)')
        plt.ylabel('Predicted Salary (₹)')
        plt.title('Actual vs Predicted Salaries')
        
        # Subplot 4: Feature Importance
        if hasattr(self.model, 'feature_importances_'):
            plt.subplot(2, 3, 4)
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.bar(range(len(importances)), importances[indices], color='lightgreen', alpha=0.7)
            plt.title('Feature Importance')
            plt.ylabel('Importance')
            plt.xticks(range(len(importances)), 
                      [self.feature_names[i] for i in indices], rotation=45)
        
        # Subplot 5: Residual Plot
        plt.subplot(2, 3, 5)
        residuals = y_test - y_pred
        plt.scatter(y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Salary (₹)')
        plt.ylabel('Residuals (₹)')
        plt.title('Residual Plot')
        
        # Subplot 6: Distribution of Residuals
        plt.subplot(2, 3, 6)
        plt.hist(residuals, bins=30, alpha=0.7, color='purple')
        plt.xlabel('Residuals (₹)')
        plt.ylabel('Frequency')
        plt.title('Distribution of Residuals')
        
        plt.tight_layout()
        plt.savefig('model_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def save_model(self, filepath):
        """Save the trained model and preprocessors"""
        model_data = {
            'model': self.model,
            'encoders': self.encoders,
            'scalers': self.scalers,
            'feature_names': self.feature_names
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to {filepath}")
    
    def predict_salary(self, job_title, experience_years, education, skills_level,
                      company_type, location, industry, work_mode):
        """Make salary prediction for new data"""
        # Create input dataframe
        input_data = pd.DataFrame({
            'job_title': [job_title],
            'experience_years': [experience_years],
            'education': [education],
            'skills_level': [skills_level],
            'company_type': [company_type],
            'location': [location],
            'industry': [industry],
            'work_mode': [work_mode]
        })
        
        # Encode categorical variables
        categorical_cols = ['job_title', 'education', 'skills_level', 'company_type', 
                           'location', 'industry', 'work_mode']
        
        for col in categorical_cols:
            if col in self.encoders:
                try:
                    input_data[col] = self.encoders[col].transform(input_data[col].astype(str))
                except ValueError:
                    # Handle unseen categories
                    input_data[col] = 0
        
        # Scale numerical features
        numerical_cols = ['experience_years']
        if 'numerical' in self.scalers:
            input_data[numerical_cols] = self.scalers['numerical'].transform(input_data[numerical_cols])
        
        # Make prediction
        prediction = self.model.predict(input_data)[0]
        
        return max(0, prediction)  # Ensure non-negative salary

def main():
    """Main training pipeline"""
    print("Employee Salary Prediction Model Training")
    print("=" * 50)
    
    # Initialize predictor
    predictor = SalaryPredictor()
    
    # Load and preprocess data
    X, y, df = predictor.load_and_preprocess_data('../data/salary_dataset.csv')
    
    # Train models
    results, X_test, y_test = predictor.train_models(X, y)
    
    # Hyperparameter tuning
    best_model = predictor.hyperparameter_tuning(X, y)
    
    # Plot results
    predictor.plot_results(results, X_test, y_test)
    
    # Save model
    predictor.save_model('../models/salary_model.pkl')
    
    # Test prediction
    print("\nTesting prediction...")
    sample_prediction = predictor.predict_salary(
        job_title='Software Engineer',
        experience_years=3,
        education='Bachelor',
        skills_level='Advanced',
        company_type='Product',
        location='Bangalore',
        industry='Product',
        work_mode='Hybrid'
    )
    
    print(f"Sample prediction: ₹{sample_prediction:,.0f}")
    
    print("\nModel training completed successfully!")

if __name__ == "__main__":
    main()
