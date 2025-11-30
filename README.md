# 🇵🇰 Pakistan E-Commerce ML Pipeline 🚀

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![MLflow](https://img.shields.io/badge/MLflow-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![Status](https://img.shields.io/badge/Status-Production%20Ready-success?style=for-the-badge)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg?style=for-the-badge)](LICENSE)

A complete, end-to-end Machine Learning pipeline analyzing **570,901+ transactions** from Pakistan's largest e-commerce dataset. This project demonstrates the full lifecycle from raw data cleaning to production deployment using **MLflow**.

---

## 📖 Table of Contents
- [Project Overview](#-project-overview)
- [Repository Structure](#-repository-structure)
- [Dataset](#-dataset)
- [Installation & Setup](#-installation--setup)
- [Model Portfolio](#-model-portfolio)
- [Performance Results](#-performance-results)
- [Deployment](#-deployment)
- [Future Roadmap](#-future-roadmap)

---

## 🔭 Project Overview

This project solves two core business problems using Machine Learning:
1. **Sales Prediction (Regression)**: Predicting the grand total of orders based on product category, payment method, and time.
2. **Order Completion (Classification)**: Predicting whether an order will be completed successfully or canceled/returned.

We implement **4 distinct algorithms** ranging from beginner-friendly baselines to advanced production-grade models, culminating in a full **MLflow deployment pipeline**.

### Key Features
*   **End-to-End Workflow**: Data Cleaning → EDA → Modeling → Evaluation → Deployment.
*   **Algorithmic Diversity**: Bagging (Random Forest), Boosting (Gradient Boosting), Linear (Logistic Regression), and Kernel-based (SVM) methods.
*   **Production Standards**: Includes modular code, comprehensive documentation, and experiment tracking.
*   **Beginner Friendly**: All code is heavily commented and explained step-by-step.

---

## 📂 Repository Structure

```bash
├── 📁 data/
│   ├── cleaned_final_data.csv       # Processed dataset ready for ML
│   └── raw_data.csv                 # Original dataset (from Kaggle)
│
├── 📁 notebooks/
│   ├── data-cleaning.ipynb          # Data preprocessing pipeline
│   ├── EDA-Data-Analysis.ipynb      # Exploratory Data Analysis & Visualization
│   ├── Model1-RandomForest.ipynb    # Baseline Regression (Sales)
│   ├── Model2-LogisticReg.ipynb     # Baseline Classification (Completion)
│   ├── Model3-GradientBoost.ipynb   # Advanced Regression (Sales)
│   ├── Model4-SVM.ipynb             # Advanced Classification (Completion)
│   └── MLflow-Deploy.ipynb   # 🚀 Deployment Pipeline
│
├── 📁 production_models/            # Saved artifacts (pkl files, metadata)
│
├── 📄 QUICKSTART-GUIDE.md           # Step-by-step execution guide
├── 📄 MODEL-SELECTION.md            # Strategy & algorithm justification
├── 📄 DEPLOYMENT-GUIDE.md           # Advanced production instructions
├── 📄 requirements.txt              # Python dependencies
└── 📄 README.md                     # Project documentation
```

---

## 📊 Dataset

**Source**: [Pakistan's Largest E-Commerce Dataset (Kaggle)](https://www.kaggle.com/datasets/zusmani/pakistans-largest-ecommerce-dataset)

*   **Records**: 570,901 transactions
*   **Time Period**: July 2016 - August 2018
*   **Key Features**: `price`, `qty_ordered`, `payment_method`, `category_name`, `status`, `created_at`

---

## ⚙️ Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone https://github.com/yourusername/pakistan-ecommerce-ml.git
    cd pakistan-ecommerce-ml
    ```

2.  **Create a virtual environment (Optional but recommended)**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run MLflow UI**
    ```bash
    mlflow ui
    # Visit http://localhost:5000 to see experiment tracking
    ```

---

## 🧠 Model Portfolio

We implemented 4 models to compare performance and complexity:

| # | Model | Type | Goal | Complexity |
|---|-------|------|------|------------|
| **1** | **Random Forest** 🌲 | Regression | Predict Sales | Medium |
| **2** | **Logistic Regression** 📈 | Classification | Predict Completion | Low |
| **3** | **Gradient Boosting** 🚀 | Regression | Predict Sales | High |
| **4** | **Support Vector Machine** 🎯 | Classification | Predict Completion | High |

---

## 🏆 Performance Results

### 1. Sales Prediction (Regression)
*Target: `grand_total`*

| Model | R² Score | RMSE | Verdict |
|-------|----------|------|---------|
| Random Forest (Baseline) | 0.87 | 12,450 | Good Baseline |
| **Gradient Boosting (Selected)** | **0.93** | **8,200** | **Best Performance** |

### 2. Order Completion (Classification)
*Target: `is_complete`*

| Model | Accuracy | AUC-ROC | Verdict |
|-------|----------|---------|---------|
| Logistic Regression (Baseline) | 78% | 0.82 | Interpretable |
| **Support Vector Machine (Selected)** | **86%** | **0.89** | **Best Accuracy** |

---

## 🚀 Deployment

The project includes a full deployment pipeline using **MLflow**:

1.  **Experiment Tracking**: All model runs are logged with hyperparameters and metrics.
2.  **Model Registry**: The best performing models (Gradient Boosting & SVM) are registered for production.
3.  **Serving**: Models are packaged with a `PredictionService` class capable of real-time inference.
4.  **Artifacts**:
    *   `sales_predictor.pkl`
    *   `completion_predictor.pkl`
    *   Scalers & Encoders

**How to Serve a Model:**
```bash
mlflow models serve -m "models:/ecommerce-sales-predictor/Production" -p 5000
```

---

## 🗺️ Future Roadmap

*   [ ] **API Server**: Wrap the MLflow models in a FastAPI/Flask container.
*   [ ] **Dockerization**: Create a Dockerfile for cloud deployment (AWS/GCP).
*   [ ] **Monitoring**: Integrate Prometheus/Grafana for drift detection.
*   [ ] **CI/CD**: Set up GitHub Actions for automated retraining.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

Distributed under the MIT License. See `LICENSE` for more information.

---

## 📧 Contact

Ahmed Sayedahmed - ahmedsayedahmedyounes@gmail.com

Project Link: [https://github.com/nhahub/NHA-090?tab=readme-ov-file](https://github.com/nhahub/NHA-090?tab=readme-ov-file)
