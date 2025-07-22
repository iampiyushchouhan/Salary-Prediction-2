#!/usr/bin/env python3
"""
Employee Salary Prediction API Script
This script loads the trained model and provides prediction functionality
"""

import pandas as pd
import numpy as np
import pickle
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SalaryPredictionAPI:
    def __init__(self, model_path):
        """Initialize the prediction API with trained model"""
        self.model_data = None
        self.load_model(model_path)
        
    def load_model(self, model_path):
        """Load the trained model and preprocessors"""
        try:
            with open(model_path, 'rb') as f:
                self.model_data = pickle.load(f)
            logger.info("Model loaded successfully")
        except FileNotFoundError:
            logger.error(f"Model file not found: {model_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
    
    def preprocess_input(self, input_data):
        """Preprocess input data for prediction"""
        # Create DataFrame
        df = pd.DataFrame([input_data])
        
        # Encode categorical variables
        categorical_cols = ['job_title', 'education', 'skills_level', 'company_type', 
                           'location', 'industry', 'work_mode']
        
        for col in categorical_cols:
            if col in self.model_data['encoders']:
                try:
                    df[col] = self.model_data['encoders'][col].transform(df[col].astype(str))
                except ValueError:
                    # Handle unseen categories by assigning most common value
                    logger.warning(f"Unseen category in {col}: {df[col].iloc[0]}")
                    df[col] = 0
        
        # Scale numerical features
        numerical_cols = ['experience_years']
        if 'numerical' in self.model_data['scalers']:
            df[numerical_cols] = self.model_data['scalers']['numerical'].transform(df[numerical_cols])
        
        return df
    
    def predict(self, input_data):
        """Make salary prediction"""
        try:
            # Preprocess input
            processed_data = self.preprocess_input(input_data)
            
            # Make prediction
            prediction = self.model_data['model'].predict(processed_data)[0]
            
            # Ensure non-negative prediction
            prediction = max(0, prediction)
            
            # Calculate confidence interval (±15%)
            confidence_interval = {
                'min': prediction * 0.85,
                'max': prediction * 1.15
            }
            
            return {
                'predicted_salary': float(prediction),
                'confidence_interval': confidence_interval,
                'currency': 'INR',
                'status': 'success'
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            return {
                'error': str(e),
                'status': 'error'
            }
    
    def get_feature_importance(self):
        """Get feature importance from the model"""
        if hasattr(self.model_data['model'], 'feature_importances_'):
            importances = self.model_data['model'].feature_importances_
            feature_names = self.model_data['feature_names']
            
            # Create feature importance dictionary
            importance_dict = dict(zip(feature_names, importances))
            
            # Sort by importance
            sorted_features = sorted(importance_dict.items(), 
                                   key=lambda x: x[1], reverse=True)
            
            return dict(sorted_features)
        else:
            return {}

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize prediction API
try:
    predictor = SalaryPredictionAPI('../models/salary_model.pkl')
except:
    logger.warning("Model not found. Starting without prediction capability.")
    predictor = None

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor is not None
    })

@app.route('/predict', methods=['POST'])
def predict_salary():
    """Salary prediction endpoint"""
    if not predictor:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        # Get input data
        input_data = request.json
        
        # Validate required fields
        required_fields = ['job_title', 'experience_years', 'education', 'skills_level',
                          'company_type', 'location', 'industry', 'work_mode']
        
        for field in required_fields:
            if field not in input_data:
                return jsonify({
                    'error': f'Missing required field: {field}',
                    'status': 'error'
                }), 400
        
        # Make prediction
        result = predictor.predict(input_data)
        
        if result['status'] == 'success':
            return jsonify(result)
        else:
            return jsonify(result), 400
            
    except Exception as e:
        logger.error(f"API error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/feature-importance', methods=['GET'])
def get_feature_importance():
    """Get feature importance endpoint"""
    if not predictor:
        return jsonify({
            'error': 'Model not loaded',
            'status': 'error'
        }), 500
    
    try:
        importance = predictor.get_feature_importance()
        return jsonify({
            'feature_importance': importance,
            'status': 'success'
        })
    except Exception as e:
        logger.error(f"Feature importance error: {str(e)}")
        return jsonify({
            'error': str(e),
            'status': 'error'
        }), 500

@app.route('/salary-ranges', methods=['GET'])
def get_salary_ranges():
    """Get salary ranges by different categories"""
    # This would typically query the database for actual statistics
    # For now, returning sample data
    sample_ranges = {
        'by_job_title': {
            'Software Engineer': {'min': 400000, 'max': 1200000, 'median': 700000},
            'Senior Software Engineer': {'min': 800000, 'max': 2500000, 'median': 1500000},
            'Data Scientist': {'min': 600000, 'max': 2000000, 'median': 1200000},
            'Product Manager': {'min': 800000, 'max': 2800000, 'median': 1800000}
        },
        'by_experience': {
            '0-2 years': {'min': 350000, 'max': 800000, 'median': 550000},
            '2-5 years': {'min': 600000, 'max': 1500000, 'median': 950000},
            '5-10 years': {'min': 1200000, 'max': 2800000, 'median': 1800000},
            '10+ years': {'min': 2000000, 'max': 3500000, 'median': 2500000}
        },
        'by_location': {
            'Bangalore': {'min': 450000, 'max': 3200000, 'median': 1200000},
            'Mumbai': {'min': 500000, 'max': 3100000, 'median': 1300000},
            'Delhi': {'min': 480000, 'max': 2800000, 'median': 1250000},
            'Hyderabad': {'min': 400000, 'max': 2000000, 'median': 850000}
        }
    }
    
    return jsonify({
        'salary_ranges': sample_ranges,
        'currency': 'INR',
        'status': 'success'
    })

def command_line_prediction():
    """Command line interface for salary prediction"""
    print("\nEmployee Salary Prediction Tool")
    print("=" * 40)
    
    if not predictor:
        print("Error: Model not loaded!")
        return
    
    # Get input from user
    print("Please enter the following details:")
    
    job_title = input("Job Title (e.g., Software Engineer): ").strip()
    experience_years = float(input("Years of Experience: "))
    education = input("Education Level (Bachelor/Master/PhD): ").strip()
    skills_level = input("Skills Level (Beginner/Intermediate/Advanced/Expert): ").strip()
    company_type = input("Company Type (IT Services/Product/Startup): ").strip()
    location = input("Location (Bangalore/Mumbai/Delhi/etc.): ").strip()
    industry = input("Industry (IT Services/Product/Fintech/etc.): ").strip()
    work_mode = input("Work Mode (Office/Remote/Hybrid): ").strip()
    
    # Prepare input data
    input_data = {
        'job_title': job_title,
        'experience_years': experience_years,
        'education': education,
        'skills_level': skills_level,
        'company_type': company_type,
        'location': location,
        'industry': industry,
        'work_mode': work_mode
    }
    
    # Make prediction
    result = predictor.predict(input_data)
    
    if result['status'] == 'success':
        prediction = result['predicted_salary']
        min_salary = result['confidence_interval']['min']
        max_salary = result['confidence_interval']['max']
        
        print("\n" + "=" * 40)
        print("SALARY PREDICTION RESULT")
        print("=" * 40)
        print(f"Predicted Salary: ₹{prediction:,.0f}")
        print(f"Expected Range: ₹{min_salary:,.0f} - ₹{max_salary:,.0f}")
        print(f"Role: {job_title} with {experience_years} years experience")
        print(f"Location: {location} ({company_type})")
        print("=" * 40)
    else:
        print(f"Prediction failed: {result.get('error', 'Unknown error')}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'api':
        # Start Flask API server
        print("Starting Salary Prediction API...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    else:
        # Run command line interface
        command_line_prediction()