# 🧫 Machine Learning for Pathogenic Biofilm Prediction in Food Environments

## 📌 Project Overview

This project focuses on the **statistical analysis, visualization, and predictive modeling of pathogenic biofilm formation in food environments** using microbiological data and Machine Learning techniques.

The project combines **data science, statistical analysis, Machine Learning, Bayesian analysis, and interactive visualization** to identify patterns associated with biofilm formation and provide a data-driven tool for prediction and risk analysis.

An interactive **Streamlit dashboard** was developed to centralize the different analytical components and allow users to explore the microbiological data, evaluate Machine Learning models, analyze risk, and perform predictions.

---

## 🎯 Project Objectives

The main objectives of this project are to:

* Analyze microbiological data related to biofilm formation.
* Perform data cleaning and preprocessing.
* Explore the statistical characteristics of the dataset.
* Study relationships between microbiological variables.
* Analyze the behavior of bacterial strains and types.
* Investigate the relationship between **Growth, OD and biofilm formation**.
* Apply statistical tests such as **ANOVA**.
* Develop Machine Learning models for predictive analysis.
* Evaluate classification performance using several metrics.
* Identify the most influential predictive variables.
* Apply **SHAP** for model interpretability.
* Perform Bayesian probability and risk analysis.
* Develop an interactive dashboard using **Streamlit**.

---

# 🧪 Data Analysis

The application provides several analytical components for exploring the microbiological dataset.

## 1. Data Loading

The dashboard allows the user to upload a CSV dataset directly through the Streamlit interface.

The application then performs the necessary preparation of the data before starting the different analyses.

---

## 2. Data Cleaning & Preprocessing

The preprocessing stage includes operations required to prepare the microbiological data for statistical analysis and Machine Learning.

The application handles:

* Data loading
* Data inspection
* Missing-value analysis
* Data cleaning
* Numerical variable processing
* Categorical variable encoding when required
* Preparation of datasets for Machine Learning
* Separation of explanatory variables and target variables

---

# 📊 Exploratory Data Analysis

The dashboard provides descriptive and graphical analysis of the microbiological dataset.

The exploratory analysis includes:

* Dataset overview
* Descriptive statistics
* Distribution analysis
* Variable exploration
* Growth analysis
* OD analysis
* Strain analysis
* Type analysis
* Correlation analysis
* Interactive visualizations

The objective is to understand the structure of the data and identify relevant patterns before applying predictive models.

---

# 📈 Statistical Analysis

Statistical analysis is integrated into the application to complement the Machine Learning approach.

## ANOVA Analysis

The application performs an **Analysis of Variance (ANOVA)** to investigate differences in Growth between groups.

The analysis provides:

* F-statistic
* p-value
* Interpretation of group differences

This allows the project to assess whether observed differences in Growth between bacterial groups are statistically significant.

> **Note:** The statistical analyses identify observed associations and differences in the available data. They should not be interpreted as proof of causal relationships.

---

# 🤖 Machine Learning

The project incorporates Machine Learning techniques for predictive analysis of biofilm formation.

## Random Forest Classification

A **Random Forest Classifier** is implemented in the application.

The model is configured using:

```python
RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
```

The model is trained directly within the Streamlit application using the prepared dataset.

Random Forest was selected because it is well suited to capturing **non-linear relationships** and interactions between variables.

---

## K-Nearest Neighbors

The project also includes **K-Nearest Neighbors (KNN)** as a Machine Learning approach for classification analysis.

KNN provides a complementary approach based on the similarity between observations.

---

## Gaussian Naive Bayes

A **Gaussian Naive Bayes** model is also integrated into the analytical workflow.

This probabilistic model is used to estimate class probabilities under the Gaussian assumption for the numerical variables.

---

# 📋 Model Evaluation

The Machine Learning component includes several evaluation tools to assess predictive performance.

The dashboard provides:

* Accuracy
* Precision
* Recall
* F1-score
* Classification report
* Confusion matrix
* Prediction probabilities

These metrics provide a more complete evaluation than accuracy alone, particularly when analyzing different biofilm-related classes.

---

# 🔍 Feature Importance

The Random Forest model provides feature importance analysis.

This allows the application to identify variables that contribute most strongly to the model's predictions.

Feature importance is displayed through visualizations in the dashboard.

This analysis helps answer questions such as:

* Which variables are most informative for prediction?
* Which microbiological measurements contribute most to model decisions?
* Which variables could deserve further investigation?

---

# 🧠 Model Interpretability with SHAP

The project integrates **SHAP (SHapley Additive exPlanations)** to improve Machine Learning interpretability.

SHAP is used to analyze the contribution of individual features to model predictions.

The objective is not only to predict biofilm-related outcomes but also to better understand **why the model produces a particular prediction**.

This provides an additional interpretability layer between the Machine Learning model and the underlying microbiological variables.

---

# 🧮 Bayesian Analysis

In addition to Machine Learning, the application includes a Bayesian analysis component.

The dashboard uses Bayesian reasoning to explore probabilities associated with biofilm formation.

The analysis includes:

* Prior probabilities
* Conditional probabilities
* Posterior probabilities
* Risk estimation
* Bayesian probability calculations

This probabilistic approach complements the predictive Machine Learning models by providing an alternative way to reason about uncertainty and risk.

---

# 🔗 Bayesian Dependency Analysis

The application also includes a **Bayesian dependency network** to represent probabilistic relationships between variables.

This component provides a visual representation of dependencies and conditional relationships within the analyzed data.

It is used as an exploratory probabilistic analysis rather than as proof of biological causality.

---

# ⚠️ Biofilm Risk Analysis

The dashboard includes a dedicated risk-analysis component.

It combines the available microbiological information and model outputs to provide an interpretable assessment of biofilm-related risk.

The risk analysis aims to transform analytical results into information that can support:

* Risk identification
* Data interpretation
* Comparative analysis
* Decision support

---

# 🎛️ Interactive Prediction

The Streamlit interface allows users to interact with the predictive component.

Users can provide the required input variables and obtain:

* Predicted class
* Prediction probability
* Model-based risk information

This transforms the Machine Learning workflow into an interactive analytical application rather than a static notebook.

---

# 📊 Sensitivity Analysis

The application includes a sensitivity-analysis component to investigate how changes in input variables can affect predictive results.

This allows the user to explore the behavior of the model under different input conditions.

Sensitivity analysis provides an additional perspective on model robustness and variable influence.

---

# 🖥️ Interactive Dashboard

The entire analytical workflow is integrated into an interactive **Streamlit dashboard**.

The dashboard brings together:

### 📁 Data Analysis

* Dataset exploration
* Data cleaning
* Descriptive statistics

### 📊 Visualization

* Distributions
* Growth analysis
* OD analysis
* Strain and type analysis
* Correlation visualization
* Interactive plots

### 📐 Statistical Analysis

* ANOVA
* Group comparison
* Statistical interpretation

### 🤖 Machine Learning

* Random Forest
* KNN
* Gaussian Naive Bayes
* Classification evaluation
* Confusion matrix
* Feature importance

### 🧠 Explainable AI

* SHAP analysis
* Feature contribution
* Model interpretation

### 🔮 Probabilistic Analysis

* Bayesian probabilities
* Risk analysis
* Dependency network

### 🎯 Prediction

* Interactive prediction
* Prediction probabilities
* Sensitivity analysis

---

# 🏗️ Project Architecture

The current project is intentionally lightweight and centered around a Streamlit application.

```text
biofilm-prediction-ml/
│
├── new2.py
│   └── Main Streamlit application
│
├── requirements.txt
│   └── Python dependencies
│
├── README.md
│   └── Project documentation
│
└── dashboard.png
    └── Dashboard preview
```

---

# 🛠️ Technologies & Libraries

## Programming Language

* **Python**

## Data Analysis

* **Pandas**
* **NumPy**

## Statistical Analysis

* **SciPy**
* Statistical tests
* ANOVA
* Correlation analysis
* Bayesian probability analysis

## Machine Learning

* **Scikit-learn**
* Random Forest
* K-Nearest Neighbors
* Gaussian Naive Bayes

## Data Visualization

* **Matplotlib**
* **Seaborn**
* **Plotly**

## Explainable AI

* **SHAP**

## Web Application

* **Streamlit**

## Model / Data Utilities

* **Joblib**

---

# 🔄 Methodology

The project follows a structured Data Science workflow:

```text
Microbiological Dataset
        │
        ▼
Data Loading
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Exploratory Data Analysis
        │
        ├──────────────► Statistical Analysis
        │
        ▼
Feature Preparation
        │
        ▼
Machine Learning
        │
        ├── Random Forest
        ├── KNN
        └── Gaussian Naive Bayes
        │
        ▼
Model Evaluation
        │
        ├── Accuracy
        ├── Precision
        ├── Recall
        ├── F1-score
        └── Confusion Matrix
        │
        ▼
Interpretability
        │
        ├── Feature Importance
        └── SHAP
        │
        ▼
Bayesian & Risk Analysis
        │
        ▼
Interactive Prediction
        │
        ▼
Streamlit Dashboard
```

---

# 📂 Application Workflow

The application follows the following analytical sequence:

1. **Upload the microbiological dataset**
2. **Inspect and clean the data**
3. **Explore descriptive statistics**
4. **Visualize microbiological variables**
5. **Analyze Growth, OD, strain and type**
6. **Perform correlation analysis**
7. **Perform ANOVA**
8. **Prepare data for Machine Learning**
9. **Train predictive models**
10. **Evaluate model performance**
11. **Analyze feature importance**
12. **Interpret predictions using SHAP**
13. **Perform Bayesian probability analysis**
14. **Estimate biofilm-related risk**
15. **Perform sensitivity analysis**
16. **Generate interactive predictions**

---

# 🚀 Installation

## 1. Clone the repository

```bash
git clone https://github.com/bdiouimira48-stack/biofilm-prediction-ml.git
```

## 2. Navigate to the project directory

```bash
cd biofilm-prediction-ml
```

## 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

---

# ▶️ Run the Application

Launch the Streamlit dashboard using:

```bash
python -m streamlit run new2.py
```

After launching the application, Streamlit will provide a local address that can be opened in a web browser.

---

# 📸 Dashboard Preview

The project includes an interactive dashboard developed with Streamlit.

![Biofilm Prediction Dashboard](dashboard.png)

---

# 🔬 Scientific Context

Biofilms are structured microbial communities that can develop on surfaces and represent an important concern in food-processing environments.

Understanding the factors associated with biofilm formation can contribute to:

* Better microbiological monitoring
* Early identification of potential risks
* Data-driven analysis of experimental results
* Improved understanding of microbial behavior
* Development of predictive analytical tools

This project explores how **statistical methods, Machine Learning and probabilistic analysis** can be combined to analyze microbiological data and support biofilm-related risk assessment.

---

# ⚠️ Scientific & Technical Limitations

This project is an academic Data Science and Machine Learning application.

The results should be interpreted within the limitations of the available dataset and modeling approach.

In particular:

* Machine Learning predictions depend on the quality and representativeness of the dataset.
* Predictive associations do not automatically imply biological causality.
* Feature importance should not be interpreted as direct biological causation.
* Bayesian results depend on the assumptions and probabilities used in the analysis.
* Model performance may change when evaluated on different datasets.
* Further external validation would be required before using the model in an operational or clinical decision-making context.

---

# 🎓 Academic Context

This project was developed as part of a final-year academic project in **Applied Mathematics and Statistics**, with a focus on:

**Data Analysis · Statistics · Machine Learning · Artificial Intelligence · Microbiological Data**

The project combines mathematical and statistical foundations with practical Machine Learning and application development.

---

# 👩‍💻 Author

**Mira Bdioui**

🎓 Applied Mathematics & Statistics
🤖 Data Science | Machine Learning | Artificial Intelligence
📍 Tunisia

GitHub: [bdiouimira48-stack](https://github.com/bdiouimira48-stack?utm_source=chatgpt.com)

---

# ⭐ Key Skills Demonstrated

This project demonstrates practical experience in:

* Python Programming
* Data Cleaning
* Exploratory Data Analysis
* Statistical Analysis
* Data Visualization
* Machine Learning
* Classification
* Random Forest
* K-Nearest Neighbors
* Gaussian Naive Bayes
* Model Evaluation
* Feature Importance
* Explainable AI
* SHAP
* Bayesian Analysis
* Risk Analysis
* Sensitivity Analysis
* Streamlit Development
* Interactive Data Applications

---

# 📌 Project Status

**Status:** Completed Academic Project

The project can be further extended with:

* Additional Machine Learning models
* Hyperparameter optimization
* Cross-validation
* External model validation
* Model persistence and deployment
* Automated data pipelines
* Cloud deployment
* Advanced Explainable AI techniques

---

# 📄 License

This repository is primarily intended for **academic and educational purposes**.

Please contact the author before using the project for commercial purposes.
