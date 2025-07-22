// Employee Salary Predictor - JavaScript Logic

// Salary prediction data and algorithms
const salaryData = {
    jobTitles: {
        "Software Engineer": { base: 600000, multiplier: 1.0 },
        "Senior Software Engineer": { base: 1200000, multiplier: 1.3 },
        "Full Stack Developer": { base: 700000, multiplier: 1.1 },
        "Data Scientist": { base: 800000, multiplier: 1.2 },
        "Product Manager": { base: 1000000, multiplier: 1.4 },
        "DevOps Engineer": { base: 750000, multiplier: 1.15 },
        "UI/UX Designer": { base: 550000, multiplier: 0.9 },
        "QA Engineer": { base: 500000, multiplier: 0.8 },
        "Business Analyst": { base: 650000, multiplier: 0.95 },
        "Project Manager": { base: 900000, multiplier: 1.25 }
    },
    companies: {
        "Google": 2.5, "Microsoft": 2.3, "Amazon": 2.2,
        "Infosys": 1.0, "TCS": 0.9, "Wipro": 0.95,
        "Accenture": 1.1, "IBM": 1.2, "Cognizant": 0.98,
        "Startup": 0.8, "Mid-size": 1.0, "Large": 1.3
    },
    locations: {
        "Bangalore": 1.2, "Hyderabad": 1.0, "Pune": 1.1,
        "Chennai": 1.0, "Mumbai": 1.3, "Delhi": 1.25,
        "Kolkata": 0.8, "Ahmedabad": 0.9, "Remote": 0.9
    },
    industries: {
        "IT Services": 1.0, "Product": 1.4, "Fintech": 1.3,
        "E-commerce": 1.2, "Healthcare": 1.1, "Education": 0.9,
        "Manufacturing": 0.95, "Consulting": 1.15
    },
    education: {
        "Diploma": 0.8, "Bachelor": 1.0, "Master": 1.2,
        "PhD": 1.4, "Certification": 0.9
    },
    skills: {
        "Beginner": 0.8, "Intermediate": 1.0, "Advanced": 1.3, "Expert": 1.6
    },
    workMode: {
        "Office": 1.0, "Remote": 0.95, "Hybrid": 1.05
    }
};

/**
 * Toggle custom company input field
 */
function toggleCustomCompany() {
    const companySelect = document.getElementById('company');
    const customDiv = document.getElementById('customCompany');
    
    if (companySelect.value === 'custom') {
        customDiv.style.display = 'block';
    } else {
        customDiv.style.display = 'none';
    }
}

/**
 * Calculate salary based on multiple factors
 * @param {Object} formData - Form input data
 * @returns {Object} Salary prediction with range
 */
function calculateSalary(formData) {
    const jobData = salaryData.jobTitles[formData.jobTitle];
    if (!jobData) return null;

    let baseSalary = jobData.base;
    let totalMultiplier = jobData.multiplier;

    // Experience factor (exponential growth)
    const expFactor = 1 + (formData.experience * 0.08) + Math.pow(formData.experience * 0.02, 2);
    totalMultiplier *= expFactor;

    // Company factor
    const companyFactor = salaryData.companies[formData.company] || 1.0;
    totalMultiplier *= companyFactor;

    // Location factor
    const locationFactor = salaryData.locations[formData.location] || 1.0;
    totalMultiplier *= locationFactor;

    // Industry factor
    const industryFactor = salaryData.industries[formData.industry] || 1.0;
    totalMultiplier *= industryFactor;

    // Education factor
    const educationFactor = salaryData.education[formData.education] || 1.0;
    totalMultiplier *= educationFactor;

    // Skills factor
    const skillsFactor = salaryData.skills[formData.skills] || 1.0;
    totalMultiplier *= skillsFactor;

    // Work mode factor
    const workModeFactor = salaryData.workMode[formData.workMode] || 1.0;
    totalMultiplier *= workModeFactor;

    // Random variation (±10%)
    const variation = 0.9 + (Math.random() * 0.2);
    totalMultiplier *= variation;

    const predictedSalary = Math.round(baseSalary * totalMultiplier);
    
    return {
        salary: predictedSalary,
        range: {
            min: Math.round(predictedSalary * 0.85),
            max: Math.round(predictedSalary * 1.15)
        },
        factors: {
            experience: expFactor,
            company: companyFactor,
            location: locationFactor,
            industry: industryFactor,
            education: educationFactor,
            skills: skillsFactor,
            workMode: workModeFactor
        }
    };
}

/**
 * Format salary amount for display
 * @param {number} amount - Salary amount
 * @returns {string} Formatted salary string
 */
function formatSalary(amount) {
    if (amount >= 10000000) {
        return `₹${(amount / 10000000).toFixed(1)} Cr`;
    } else if (amount >= 100000) {
        return `₹${(amount / 100000).toFixed(1)} L`;
    } else {
        return `₹${amount.toLocaleString()}`;
    }
}

/**
 * Validate form data
 * @param {Object} formData - Form data to validate
 * @returns {boolean} Validation result
 */
function validateFormData(formData) {
    const required = ['jobTitle', 'education', 'skills', 'company', 'location', 'industry', 'workMode'];
    
    for (let field of required) {
        if (!formData[field] || formData[field].trim() === '') {
            alert(`Please select ${field.replace(/([A-Z])/g, ' $1').toLowerCase()}`);
            return false;
        }
    }
    
    if (isNaN(formData.experience) || formData.experience < 0 || formData.experience > 50) {
        alert('Please enter valid years of experience (0-50)');
        return false;
    }
    
    return true;
}

/**
 * Log prediction for analytics
 * @param {Object} formData - Form data
 * @param {Object} prediction - Prediction result
 */
function logPrediction(formData, prediction) {
    const logData = {
        timestamp: new Date().toISOString(),
        inputs: formData,
        prediction: prediction.salary,
        range: prediction.range
    };
    
    // In a real application, you would send this to your analytics service
    console.log('Prediction Log:', logData);
}

// Main form submission handler
document.addEventListener('DOMContentLoaded', function() {
    const form = document.getElementById('salaryForm');
    const loading = document.getElementById('loading');
    const result = document.getElementById('result');
    const salaryAmount = document.getElementById('salaryAmount');
    const resultText = document.getElementById('resultText');

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        
        // Collect form data
        const formData = {
            jobTitle: document.getElementById('jobTitle').value,
            experience: parseInt(document.getElementById('experience').value) || 0,
            education: document.getElementById('education').value,
            skills: document.getElementById('skills').value,
            company: document.getElementById('company').value === 'custom' ? 
                    document.getElementById('customCompanyName').value : 
                    document.getElementById('company').value,
            location: document.getElementById('location').value,
            industry: document.getElementById('industry').value,
            workMode: document.getElementById('workMode').value
        };

        // Validate form data
        if (!validateFormData(formData)) {
            return;
        }
        
        // Show loading
        loading.style.display = 'block';
        result.style.display = 'none';
        
        // Simulate API delay for better UX
        setTimeout(() => {
            const prediction = calculateSalary(formData);
            
            if (prediction) {
                salaryAmount.textContent = formatSalary(prediction.salary);
                resultText.innerHTML = `
                    <strong>Expected Salary Range: ${formatSalary(prediction.range.min)} - ${formatSalary(prediction.range.max)}</strong><br>
                    Based on ${formData.jobTitle} role with ${formData.experience} years of experience<br>
                    at ${formData.company} in ${formData.location} (${formData.industry} industry)
                `;
                
                // Log prediction for analytics
                logPrediction(formData, prediction);
                
                loading.style.display = 'none';
                result.style.display = 'block';
                
                // Scroll to result
                result.scrollIntoView({ behavior: 'smooth' });
            } else {
                alert('Error calculating salary. Please check your inputs and try again.');
                loading.style.display = 'none';
            }
        }, 2000);
    });
});

// Export functions for testing (if in Node.js environment)
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        calculateSalary,
        formatSalary,
        validateFormData,
        salaryData
    };
}