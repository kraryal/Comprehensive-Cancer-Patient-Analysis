# 🏥 Comprehensive Cancer Patient Analysis

<div align="center">

![Python](https://img.shields.io/badge/Python-3.8+-blue?style=for-the-badge&logo=python)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?style=for-the-badge&logo=pandas)
![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-green?style=for-the-badge)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?style=for-the-badge&logo=jupyter)

</div>

---

## 📊 Overview

A **comprehensive data analysis project** focused on a global cancer patient dataset comprising **50,000 patients** across **10 countries** from **2015-2024**. This study examines demographic patterns, risk factors, treatment costs, survival outcomes, and temporal trends in cancer care using advanced statistical methods and machine learning techniques.

> 🎯 **Purpose**: Analyze global cancer patient data to identify trends, survival factors, and treatment effectiveness patterns for better clinical decision-making and healthcare policy development

---

## 🔬 Key Research Findings

### **🎯 Primary Risk Factors**
- **Smoking**: Most significant modifiable risk factor (R² = 0.235, 26.0% feature importance)
- **Genetic Risk**: Strongest predictor alongside smoking (R² = 0.229, 25.6% feature importance)
- **Air Pollution**: Moderate correlation with severity (R² = 0.135)
- **Alcohol Use**: Significant correlation with cancer severity (R² = 0.132)
- **Obesity**: Weakest but still significant factor (R² = 0.063)

### **🏥 Clinical Insights**
- **Early-Stage Diagnosis**: Consistent 39-40% rate across all cancer types
- **Survival Outcomes**: Mean survival of 5.0 years across all patients
- **Treatment Costs**: Relatively uniform globally ($51,568 - $52,899)
- **Stage Distribution**: No significant survival differences across cancer stages

### **🌍 Global Healthcare Patterns**
- **Geographic Balance**: Uniform representation across 10 countries (9.7% - 10.2% each)
- **Gender Distribution**: Balanced across Male (33.6%), Female (33.4%), Other (33.0%)
- **Temporal Trends**: Stable costs and outcomes over 2015-2024 period

---

## ✨ Key Features

### 📈 **Advanced Statistical Analysis**
- Comprehensive correlation analysis of risk factors
- Machine Learning severity prediction (Random Forest)
- Survival pattern analysis and Kaplan-Meier curves
- Economic burden assessment across countries

### 🔍 **Data Exploration & Modeling**
- 50,000 patient records with no missing data
- Multi-country comparative analysis
- Risk factor interaction modeling
- Temporal trend analysis (2015-2024)

### 📊 **Comprehensive Visualizations**
- Risk factor correlation heatmaps
- Early-stage diagnosis rate comparisons
- Economic burden analysis by demographics
- Feature importance rankings from ML models

### 🤖 **Predictive Analytics**
- Random Forest Regressor for severity prediction
- Feature importance analysis
- Model performance evaluation (Training R²: 0.969, Testing R²: 0.768)

---

## 🎯 Clinical Applications

**🔬 Risk Assessment**
- **Primary Focus**: Smoking cessation programs (highest impact factor)
- **Genetic Counseling**: Enhanced genetic risk assessment protocols
- **Environmental Health**: Air pollution exposure mitigation strategies
- **Lifestyle Interventions**: Alcohol use and obesity management programs

**🏥 Healthcare Policy**
- **Early Detection**: Target improvement of 39-40% early diagnosis rates
- **Cost Management**: Address global treatment cost disparities
- **Resource Allocation**: Optimize based on risk factor hierarchies
- **Screening Programs**: Enhanced lung cancer screening (lowest early detection at 38.4%)

---

## 🗂️ Project Structure

```
Comprehensive-Cancer-Patient-Analysis/
│
├── data/
│   ├── raw/                    # Original 50K patient dataset
│   ├── processed/              # Cleaned and processed data
│   └── analysis_results/       # Statistical analysis outputs
│
├── notebooks/
│   ├── 01_demographic_analysis.ipynb    # Population characteristics
│   ├── 02_risk_factor_analysis.ipynb    # Correlation and regression
│   ├── 03_ml_severity_prediction.ipynb  # Machine learning models
│   ├── 04_survival_analysis.ipynb       # Survival patterns
│   ├── 05_economic_analysis.ipynb       # Treatment cost analysis
│   └── 06_temporal_trends.ipynb         # 2015-2024 trends
│
├── src/
│   ├── data_processing.py      # Data cleaning utilities
│   ├── statistical_analysis.py # Risk factor correlations
│   ├── ml_models.py           # Random Forest implementation
│   ├── survival_analysis.py    # Survival curve analysis
│   └── visualization.py        # Plotting utilities
│
├── results/
│   ├── figures/                # Generated plots and charts
│   ├── reports/                # Analysis reports (14-page comprehensive)
│   ├── models/                 # Saved ML model files
│   └── statistical_outputs/    # Test results and summaries
│
├── requirements.txt            # Project dependencies
└── README.md
```

---

## ⚡ Quick Start

### 🔧 Prerequisites

**Essential Requirements:**
- Python 3.8+
- Jupyter Notebook/Lab
- Statistical analysis background
- Healthcare data familiarity (recommended)

**Key Libraries:**
```bash
pandas>=1.3.0          # Data manipulation
numpy>=1.21.0          # Numerical computing
matplotlib>=3.4.0      # Basic plotting
seaborn>=0.11.0        # Statistical visualization
scikit-learn>=1.0.0    # Machine learning
scipy>=1.7.0           # Statistical tests
statsmodels>=0.13.0    # Advanced statistics
lifelines>=0.27.0      # Survival analysis
```

### 🚀 Installation

```bash
# Clone the repository
git clone https://github.com/kraryal/Comprehensive-Cancer-Patient-Analysis.git

# Navigate to project directory
cd Comprehensive-Cancer-Patient-Analysis

# Create virtual environment
python -m venv cancer_analysis_env

# Activate virtual environment
# Windows:
cancer_analysis_env\Scripts\activate
# macOS/Linux:
source cancer_analysis_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Launch Jupyter Notebook
jupyter notebook
```

---

---

## 💡 Usage Examples

### 📊 **Basic Data Analysis**

```python

# Load the dataset
data = pd.read_csv("global_cancer_patients_2015_2024.csv")

print(f"📈 Dataset Overview:")
print(f"• Total Records: {data.shape[0]:,}")
print(f"• Total Features: {data.shape[1]}")
print(f"• Date Range: 2015-2024")
print(f"• Duplicate Records: {data.duplicated().sum()}")

# Display first few rows
print("\n🔍 Sample Data:")
display(data.head())

# Dataset information
print("\n📋 Dataset Information:")
data.info()

```


### 🤖 **Predictive Modeling**

```python
def build_severity_prediction_model():
    """Build and evaluate a model to predict cancer severity"""
    
    print("🤖 BUILDING CANCER SEVERITY PREDICTION MODEL")
    print("=" * 50)
    
    # Prepare data for machine learning
    data_ml = data.copy()
    
    # Encode categorical variables
    categorical_cols = ["Gender", "Country_Region", "Cancer_Type", "Cancer_Stage"]
    label_encoders = {}
    
    for col in categorical_cols:
        le = LabelEncoder()
        data_ml[col] = le.fit_transform(data_ml[col])
        label_encoders[col] = le
    
    # Prepare features and target
    feature_cols = ["Age", "Gender", "Country_Region", "Cancer_Type", "Cancer_Stage", 
                   "Genetic_Risk", "Air_Pollution", "Alcohol_Use", "Smoking", "Obesity_Level"]
    
    X = data_ml[feature_cols]
    y = data_ml["Target_Severity_Score"]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=200, max_depth=None, 
                                   min_samples_split=2, min_samples_leaf=1, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Evaluate model
    train_r2 = r2_score(y_train, rf_model.predict(X_train))
    test_r2 = r2_score(y_test, rf_model.predict(X_test))
    
    print(f"📊 MODEL PERFORMANCE:")
    print(f"• Training R² Score: {train_r2:.4f}")
    print(f"• Testing R² Score: {test_r2:.4f}")
    print(f"• Model Generalization: {'Good' if abs(train_r2 - test_r2) < 0.1 else 'Needs Improvement'}")
    
    # Feature importance analysis
    feature_importance = pd.Series(rf_model.feature_importances_, 
                                 index=feature_cols).sort_values(ascending=False)
    
    # Visualization
    plt.figure(figsize=(12, 8))
    colors = sns.color_palette("viridis", len(feature_importance))
    bars = plt.bar(range(len(feature_importance)), feature_importance.values, color=colors)
    
    plt.title('🎯 FEATURE IMPORTANCE FOR CANCER SEVERITY PREDICTION', fontweight='bold', fontsize=16)
    plt.xlabel('Features', fontweight='bold')
    plt.ylabel('Importance Score', fontweight='bold')
    plt.xticks(range(len(feature_importance)), 
               [name.replace('_', ' ').title() for name in feature_importance.index], rotation=45)
    
    # Add value labels on bars
    for bar, importance in zip(bars, feature_importance.values):
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.001,
                f'{importance:.3f}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n🔍 TOP PREDICTORS OF CANCER SEVERITY:")
    for i, (feature, importance) in enumerate(feature_importance.head().items(), 1):
        print(f"{i}. {feature.replace('_', ' ').title()}: {importance:.3f} ({importance*100:.1f}%)")
    
    return rf_model, feature_importance

rf_model, feature_importance = build_severity_prediction_model()
```

---

## 📊 Research Methodology

### **Statistical Approach**
1. **Descriptive Analysis**: Comprehensive demographic profiling
2. **Correlation Analysis**: Pearson and Spearman coefficients
3. **Regression Modeling**: Linear regression with interaction terms
4. **Machine Learning**: Random Forest with feature importance
5. **Hypothesis Testing**: Kruskal-Wallis tests for group comparisons
6. **Survival Analysis**: Pattern analysis across cancer stages

### **Key Statistical Tests**
- **Cost across Stages**: Kruskal-Wallis H = 3.92, p = 0.4254 (No significant difference)
- **Survival across Stages**: Kruskal-Wallis H = 2.75, p = 0.6033 (No significant difference)
- **Risk Factor Interactions**: Comprehensive correlation matrix analysis

---

## 🎯 Clinical Recommendations

### **🚨 High Priority Interventions**

1. **Enhanced Lung Cancer Screening**
   - Target: Improve 38.4% early detection rate
   - Action: Develop targeted screening programs

2. **Smoking Cessation Programs**
   - Rationale: 26% contribution to severity prediction
   - Focus: Primary modifiable risk factor

3. **Healthcare Cost Equity**
   - Goal: Address global disparities while maintaining quality
   - Range: Current $1,331 difference between countries

### **⚡ Medium Priority Actions**

1. **Genetic Risk Assessment**
   - Implementation: Comprehensive genetic counseling programs
   - Impact: 25.6% feature importance in severity prediction

2. **Predictive Model Enhancement**
   - Current: 76.8% testing accuracy
   - Target: Improve generalization and reduce overfitting

3. **Environmental Health Programs**
   - Focus: Air pollution exposure mitigation
   - Correlation: R² = 0.135 with cancer severity

---

## 📚 Research Impact & Publications

### **Key Findings Published**
- **Dataset Size**: 50,000 patients across 10 countries (2015-2024)
- **Geographic Coverage**: USA, UK, Canada, Australia, Germany, China, India, Brazil, Russia, Pakistan
- **Cancer Types**: 8 major types with uniform early detection rates
- **Temporal Stability**: Consistent outcomes over 10-year period

### **Academic References**
- World Health Organization Cancer Statistics
- CA: A Cancer Journal for Clinicians publications
- The Lancet global cancer survival studies
- BMJ landmark smoking-mortality studies

---

## ⚠️ Important Limitations

### **Study Constraints**
1. **Geographic Representation**: Limited to 10 countries
2. **Temporal Scope**: 2015-2024 may not capture longer-term outcomes
3. **Variable Standardization**: Cross-country measurement variations
4. **Missing Clinical Data**: Comorbidities and treatment specifics not included
5. **Model Overfitting**: Machine learning model requires refinement

### **Ethical Considerations**
- All patient data **anonymized** and **de-identified**
- Analysis complies with **HIPAA** and healthcare data protection standards
- Results for **research and educational purposes** only
- **Not intended for clinical decision-making** without proper validation

---

## 🤝 Contributing

We welcome contributions from healthcare professionals, data scientists, and researchers!

### **Contribution Areas**
- 🔬 Advanced statistical methods
- 📊 Enhanced visualization techniques  
- 🤖 Improved machine learning models
- 📝 Clinical interpretation and validation
- 🌍 Additional geographic datasets

### **How to Contribute**

```bash
# Fork and clone
git clone https://github.com/YOUR-USERNAME/Comprehensive-Cancer-Patient-Analysis.git

# Create feature branch
git checkout -b feature/clinical-enhancement

# Make improvements
# - Add new analysis methods
# - Improve model performance
# - Enhance visualizations

# Commit changes
git commit -m "Add advanced survival analysis methods"

# Submit pull request
```

---

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

---

<div align="center">

**Created with ❤️ by [Krishna Aryal](https://github.com/kraryal)**  
*Georgia Institute of Technology - MS Analytics*

[![GitHub followers](https://img.shields.io/github/followers/kraryal?style=social)](https://github.com/kraryal)
[![GitHub stars](https://img.shields.io/github/stars/kraryal/Comprehensive-Cancer-Patient-Analysis?style=social)](https://github.com/kraryal/Comprehensive-Cancer-Patient-Analysis)

---

### 🏥 **Advancing Healthcare Through Data Science**
**📊 50K Patients • 🌍 10 Countries • 📅 10 Years**

**Star this repo ⭐ if you found it valuable for your research!**

</div>
```
