# 💼 Employee Salary Predictor

<div align="center">
  
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Flask](https://img.shields.io/badge/Flask-2.3.2-green.svg)](https://flask.palletsprojects.com/)
[![Scikit-Learn](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Github--Repo-Active-success.svg)](https://github.com/iampiyushchouhan/Salary-Prediction.git)

</div>

---
## 🚀 AI-Powered Salary Prediction Tool for Indian Tech Industry

 ## 🎯 Overview 
The Employee Salary Predictor is an advanced AI-powered web application that predicts salary ranges for tech professionals in India. Using machine learning algorithms and comprehensive market data, it analyzes multiple factors including experience, skills, company type, location, and industry to provide accurate salary predictions.

## 🎯 Problem Statement
- Lack of transparency in salary information across the Indian tech industry
- Difficulty for professionals to negotiate fair compensation
- Need for data-driven salary insights for career planning

## 💡 Solution
- ML-powered salary prediction with 90%+ accuracy
- Real-time predictions through intuitive web interface
- Comprehensive analysis of 8+ key factors affecting salaries
- RESTful API for integration with other platforms

---

# ✨ Features

## 🔮 Smart Predictions
- **Multi-factor Analysis:** Experience, skills, location, company type, industry, education, work mode
- **Real-time Results:** Instant salary predictions with confidence intervals
- **Accuracy:** 90%+ prediction accuracy based on comprehensive training data
- **Range Estimation:** Provides minimum, maximum, and expected salary ranges

## 🎨 User Experience
- **Responsive Design:** Works seamlessly on desktop, tablet, and mobile devices
- **Intuitive Interface:** Clean, modern UI with step-by-step form guidance
- **Interactive Elements:** Smooth animations and real-time validation
- **Accessibility:** WCAG 2.1 compliant for inclusive user experience

## 🔧 Technical Features
- **RESTful API:** Complete API for programmatic access
- **Model Versioning:** Systematic model training and deployment pipeline
- **Error Handling:** Comprehensive error handling and logging
- **Performance:** Optimized for fast response times (<2 seconds)

## 📊 Analytics & Insights
- **Feature Importance:** Understanding which factors impact salary most
- **Market Trends:** Salary ranges by role, experience, and location
- **Comparative Analysis:** Benchmarking across different parameters
---

## 🛠️ Technology Stack

### 🎨 **Frontend**

- 🧱**HTML5** – Semantic markup and structure
[![HTML5](https://img.shields.io/badge/HTML5-E34F26?style=flat&logo=html5&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/Guide/HTML/HTML5)  
  

- 🎨**CSS3** – Modern styling with Flexbox/Grid
[![CSS3](https://img.shields.io/badge/CSS3-1572B6?style=flat&logo=css3&logoColor=white)](https://developer.mozilla.org/en-US/docs/Web/CSS)  
  

- ⚡**JavaScript** – Interactive functionality
[![JavaScript](https://img.shields.io/badge/JavaScript-F7DF1E?style=flat&logo=javascript&logoColor=black)](https://developer.mozilla.org/en-US/docs/Web/JavaScript)

### 🧠 Backend & Machine Learning

- 🐍**Python 3.8+** – Core programming language
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/downloads/)  
  
- 🤖**Scikit-learn** – Machine learning algorithms
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)  
  

- 🧮**Pandas** – Data manipulation and analysis
[![Pandas](https://img.shields.io/badge/pandas-150458?style=flat&logo=pandas&logoColor=white)](https://pandas.pydata.org/)  
  

- 🔢**NumPy** – Numerical computing
[![NumPy](https://img.shields.io/badge/numpy-013243?style=flat&logo=numpy&logoColor=white)](https://numpy.org/)  
  

### 📊 Data Visualization

- 📈**Matplotlib** – Statistical plotting
[![Matplotlib](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)](https://matplotlib.org/)  
  

- 🌈**Seaborn** – Advanced visualizations
[![Seaborn](https://img.shields.io/badge/Seaborn-3776AB?style=flat&logo=python&logoColor=white)](https://seaborn.pydata.org/)  
  
### 🚀 Deployment

- 🧪**Streamlit** – Interactive web app deployment
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)  
  

- 🌐**GitHub Pages** – Static site hosting
[![GitHub Pages](https://img.shields.io/badge/GitHub_Pages-222222?style=flat&logo=github&logoColor=white)](https://pages.github.com/)  

---

## 📊 Model Performance
<div align="center">

| Metric            | Value              | Description                     |
|-------------------|--------------------|---------------------------------|
| R² Score          | 0.904              | Excellent model fit (90.4%)     |
| RMSE              | ₹1,98,605          | Root Mean Square Error          |
| MAE               | ₹1,45,230          | Mean Absolute Error             |
| Cross-Validation  | 0.897 ± 0.023      | 5-fold CV performance           |



### 🤖 Model Comparison Results

| Algorithm         | R² Score           | RMSE (₹)         | Training Time  |
|-------------------|--------------------|------------------|----------------|
| 🌲 Random Forest   | 0.904              | 1,98,605         | 2.3s           |
| 🚀 Gradient Boosting | 0.886              | 2,15,309         | 8.7s           |
| 📈 Linear Regression | 0.851              | 2,46,669         | 0.1s           |
| 🧮 Ridge Regression  | 0.885              | 2,16,939         | 0.2s           |
| 🌀 SVR               | 0.678              | 3,85,976         | 15.2s          |

</div>

--- 

# 📁 Project Structure
```bash
salary-predictor/
├── 📄 index.html                 # Main web interface
├── 📁 css/
│   └── 🎨 styles.css            # Application styles
├── 📁 js/
│   └── ⚡ script.js             # Frontend logic
├── 📁 data/
│   └── 📊 salary_dataset.csv    # Training dataset
├── 📁 models/
│   └── 🤖 salary_model.pkl      # Trained ML model
├── 📁 python/
│   ├── 🧠 train_model.py        # Model training script
│   ├── 🔮 predict_salary.py     # Prediction API
│   └── 📋 requirements.txt      # Python dependencies
├── 📸 screenshots/              # UI screenshots
├── 📚 docs/                     # Additional documentation
└── 📖 README.md                 # This file
```
---
# 💻 Installation
### 1️⃣ Clone the Repository
```bash
git clone https://github.com/yourusername/salary-predictor.git
cd salary-predictor
```
### 2️⃣ Create Virtual Environment
```bash
# Create virtual environment
python -m venv salary_env

# Activate virtual environment
# On Windows:
salary_env\Scripts\activate
# On macOS/Linux:
source salary_env/bin/activate
```
### 3️⃣ Install Dependencies
```bash
pip install -r python/requirements.txt
```
### 4️⃣ Verify Installation
```bash
python python/predict_salary.py
# Should start the command-line interface
```
---
# 📈 Usage
### 🖥️ Web Interface

1.Open index.html in your browser or start the Flask server
2.Fill in your professional details:
- Job title/role
- Years of experience
- Education level
- Skill proficiency
- Company type
- Location
- Industry
- Work mode preference
Click "Predict My Salary" to get instant results
---

# 📚 API Documentation
### 🐍 Python API
```python
from python.predict_salary import SalaryPredictionAPI

# Initialize predictor
predictor = SalaryPredictionAPI('models/salary_model.pkl')

# Make prediction
result = predictor.predict({
    'job_title': 'Software Engineer',
    'experience_years': 3,
    'education': 'Bachelor',
    'skills_level': 'Advanced',
    'company_type': 'Product',
    'location': 'Bangalore',
    'industry': 'Product',
    'work_mode': 'Hybrid'
})

print(f"Predicted Salary: ₹{result['predicted_salary']:,.0f}")
```
### 🔮 Salary Prediction
```http
POST /predict
Content-Type: application/json

{
  "job_title": "Software Engineer",
  "experience_years": 3,
  "education": "Bachelor",
  "skills_level": "Advanced",
  "company_type": "Product",
  "location": "Bangalore",
  "industry": "Product",
  "work_mode": "Hybrid"
}
```
### Response:
```json
{
  "predicted_salary": 850000,
  "confidence_interval": {
    "min": 722500,
    "max": 977500
  },
  "currency": "INR",
  "status": "success"
}
```
---
