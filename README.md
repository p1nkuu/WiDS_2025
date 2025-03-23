# WiDS_2025

### Team Members 
- **[Tiffany Nguyen](https://github.com/p1nkuu)**
- **[Tia Zheng](https://github.com/tiaz26)**
- **[Seungmin Cho](https://github.com/ojnim)**
- **[Alison Ramirez](https://github.com/AliRH02)**

### **👥 Team Members**

| Name | GitHub Handle | Contribution |
| ----- | ----- | ----- |
| Tiffany Nguyen| [@Tiffany Nguyen](https://github.com/p1nkuu) |  |
|Tia Zheng| [@Tia Zheng](https://github.com/tiaz26)|  |
| Seungnim Cho| [@Seungmin Cho](https://github.com/ojnim) |  |
| Alison Ramirez| [@Alison Ramirez](https://github.com/AliRH02)|Preprocessing, EDA, Modeling, and Optimization for target Sex_F |

---
## **🎯 Project Highlights**
- Techniques Used: We applied Regression and Logistic Regression models to analyze the data.
- Model Performance: Logistic Regression outperformed Regression, achieving an initial accuracy of 0.69, which improved to 0.79 after optimization.
- Key Findings: 
  - There was a significant class difference in the dataset.
  -	No direct correlation was found between the provided data and the target variable Sex_F
    (female classification).
  - Despite the lack of direct correlation, the Logistic Regression model was able to make meaningful predictions after optimization.

🔗 [WiDS Datathon 2025 | Kaggle Competition Page](https://www.kaggle.com/competitions/widsdatathon2025/overview)

## **👩🏽‍💻 Setup & Execution**
### Requirements
To reproduce the project, ensure you have the following installed:
- Python 
- Jupyter Notebook
- Ensure dataset is available in the expected path
- Required libraries: (Found in the *Sex_FModel.ipynb*)
```bash
import re
import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
import seaborn as sns
import lightgbm as lgb
from sklearn.svm import SVC
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.under_sampling import RandomUnderSampler
from sklearn.metrics import precision_recall_curve, auc, classification_report
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
```
### Project Files
The repository contains the following files:
-	*Preprocessing.ipynb* – All of the Data preprocessed 
-	*Exploratory Data Analysis.ipynb*– Data Exploration Analysis for target Sex_F
-	*second_submission.csv* – Final predictions submitted to the competition.
-	*Logistic_regression_secondsubmissiong_model.pkl* – Trained Logistic Regression model for predicting Sex_F.
-	*Sex_FModel.ipynb* – Jupyter Notebook with the complete training, evaluation, and optimization process.

## Steps to Run the Model
1.	Clone the Repository 
```bash
git clone https://github.com/p1nkuu/WiDS_2025
```
2.	Navigate to the Project Folder:
```bash
cd WiDS_2025
```
3.	Open the Jupyter Notebook
```bash
jupyter notebook Sex_FModel.ipynb
```
4.	Run the Notebook Cells
   - The notebook contains steps for: 
     - Model training with Logistic Regression
     - Evaluation (AUC, precision-recall, accuracy improvements)
  - For Data Preprocessing and Exploratory Analysis
      - Open the *Preprocessing.ipynb* or
      - Open the *Exploratory Data Analysis.ipynb*
5.	Load & Use the Pre-trained Model *(Optional)*:
If you want to load the trained model and make predictions: use the file 
```bash
import joblib  

model = joblib.load("logistic_regression_secondsubmission_model.pkl")  

# Example prediction (replace [...] with actual feature values)
sample_data = np.array([...]).reshape(1, -1)  
prediction = model.predict(sample_data)  
prediction_proba = model.predict_proba(sample_data)[:, 1]  # Probability score

print("Predicted Class:", prediction)
print("Prediction Probability:", prediction_proba)
```
## **🏗️ Project Overview**
The **WiDs Datathon Global Challenge** Kaggle competition is an opportunity for BreakThrough Tech AI student fellows to apply and enhance their data science skills while working on an impactful and important problem. The BreakThrough Tech AI Program provides students with a strong foundation in machine learning, equipping them with the skills needed to tackle real-world challenges like the one presented in this competition.
### Goal of the Challenge 
The goal of his challenge is to build a model using **fMRI data** and **Socio-demographic information** to predict:
1. Individual's sex (Male or Female)
2. Individual's ADHD diagnosis
By analyzing the data, participants aim to uncover patterns and insights that contribute to a better understanding of ADHD and reveal potential biases in machine learning models used in this field.

### Real World Significance and Impact
ADHD is one of the most common neurodevelopmental disorders, affecting approximately **11% of adolescents**. However, its manifestation varies significantly between **males and females**, with females often being underdiagnosed due to differences in symptom presentation. Undiagnosed ADHD can lead to long-term mental health struggles, making it difficult for individuals to access the resources and support they need to thrive.

## **📊 Data Exploration**
The Wids Datathon competition leverages datasets provided by the **Healthy Brain Network (HBN)**, the signature scientific initiative of the **Child Mind Institute**, and the *Reproducible Brain Chrats project(RBC)**. Insights from this competition could help improve ADHD diagnosis, reduce gender biases in treatment, and advance research in neurodevelopmental disorders. The goal of the project is to predict an individual’s sex (Sex_F) and ADHD diagnosis (ADHD_Outcome) using these features.
The dataset is structured as follows:
- **Training Data (TRAIN_OLD):** *(TRAINING_SOLUTIONS.xlsl)*
  
  1.**Functional MRI Data:** *(TRAIN_FUNCTIONAL_CONNECTOME_MATRICES.csv)*: Connectome matrices that describe brain activity relationships.
  
  2.**Socio-Demographic & Parenting Data:** *(TRAIN_CATEGORICAL_METADATA.xlsl & TRAIN_QUANTITATIVE_METADATA.xlsl)*: Information about ethnicity, family,
     emotions (Strength and Difficulties Questionnaire), and parenting (Alabama Parenting Questionnaire).
  
- **Test Data (TEST)**: Contains the same structure as the training data but without the target labels.Used to evaluate model performance.
    1.**Functional MRI Data:** *(TEST_FUNCTIONAL_CONNECTOME_MATRICES.csv)*

    2.**Socio-Demographic & Parenting Data:** *(TEST_CATEGORICAL.xlsl & TEST_QUANTITATIVE_METADATA.xlsl)*

### Data Preprocessing
Before applying machine learning models, we performed several preprocessing steps:

1.Handling Categorical Data
- One-hot encoding was applied to categorical features:
```bash
train_categorical_encoded = pd.get_dummies(train_categorical, 
   columns=['Basic_Demos_Study_Site', 'PreInt_Demos_Fam_Child_Ethnicity', 'MRI_Track_Scan_Location'], 
   drop_first=False)
```
- Boolean values were converted to integers for consistency:
```bash
train_categorical_encoded = train_categorical_encoded.apply(lambda x: x.astype(int) if x.dtype == 'bool' else x)
```
2.Handling Missing Values
- Used K-Nearest Neighbors Imputation (KNNImputer) to fill missing numerical values:
```bash
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5)
train_quantitative_numeric = train_quantitative.drop(columns=['participant_id'])
train_quantitative_imputed = imputer.fit_transform(train_quantitative_numeric)
train_quantitative[train_quantitative_numeric.columns] = train_quantitative_imputed
```
3.Feature Correlation Analysis
- We examined the correlation of numerical features with MRI_Track_Age_at_Scan to understand key relationships:
```bash
corr_matrix = train_quantitative_numeric.corr()
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix[['MRI_Track_Age_at_Scan']].sort_values(by='MRI_Track_Age_at_Scan', ascending=False), 
            annot=True, cmap='coolwarm', vmin=-1, vmax=1)
plt.title("Correlation with MRI_Track_Age_at_Scan")
plt.show()
```
4.Merging Data: 
- We merged all of the preprocessed Data into one dataframe converted into a .pkl:
```bash
train_categorical.to_pickle("proccessed_categorical_data.pkl")
```

### Exploratory Data Analysis (EDA)

1.Correlation Matrix of Quantitative Features: Sex_F

To understand the relationships between numerical features, a correlation matrix was computed and visualized using a heatmap:
![Quantitative_Correlation_Matrix](https://github.com/user-attachments/assets/7c8968ce-d233-49df-a2bb-ec53ef23421c)

2.Target Variable Distribution: Sex_F

Examining the distribution of the target variable Sex_F:
![Target variable Sex_F](https://github.com/user-attachments/assets/73b628d2-7eda-4ae7-8b67-41d4434154d1)

3.Principal Component Analysis (PCA): Sex_F

Principal Component Analysis (PCA) was performed to reduce the dataset's dimensionality while preserving as much variance as possible, allowing for better visualization and interpretation of key patterns in the data.
![Screenshot 2025-03-21 163533](https://github.com/user-attachments/assets/0b2c48e7-716a-43e7-92ac-0d26647be50b)

## **🧠 Model Development**
To classify the target variable "Sex_F," a Random Forest classifier and a Logistic Regression model were developed. The dataset was first split into training and test sets using an 80-20 split to ensure effective model training and evaluation.

### Model Selection and Training Approach
  1. Random Forest Classifier:
     - The initial Random Forest model was trained using default parameters and achieved an accuracy of **69.55%**. However, the classification report indicated poor performance for the minority class (Sex_F = 1), with a recall of **0.00**, suggesting significant class imbalance.
      - To address this issue, Synthetic Minority Over-sampling Technique (SMOTE) was applied to balance the dataset before retraining.
      - The Random Forest model was then re-trained with class_weight='balanced' to further mitigate imbalance issues.

  3. Logistic Regression:
     - Logistic Regression with class_weight='balanced' was trained as an alternative approach.
     - The model achieved an accuracy of **61%**, with better recall for the minority class compared to the initial Random Forest model.
     - Additional evaluation metrics included the **ROC-AUC score (0.6258)** and **PR-AUC score (0.4375)**, providing insights into the model’s performance on imbalanced data.
       
### Hyperparameter Tuning
Hyperparameter tuning was performed using RandomizedSearchCV for Random Forest and GridSearchCV for Logistic Regression:
  1. Random Forest Optimization:
     - n_estimators tuned between 100 and 300 (optimal: 150).
     - max_depth options: 10, 20, None (optimal: None).
     - min_samples_split options: 2, 5, 10 (optimal: 2).
     - class_weight set to 'balanced'.
     - The optimized model improved performance slightly but still struggled with class imbalance.
  2. Logistic Regression Optimization:
     - C (regularization strength): 0.1, 1, 10 (optimal: 10).
     - solver: 'lbfgs', 'liblinear' (optimal: 'liblinear').
     - max_iter: 100, 200 (optimal: 100).
     - The tuned Logistic Regression model achieved an improved accuracy of 79.01%.
  
## **📈 Results & Key Findings**
### Model Performance Evaluation
  - The Random Forest classifier (after balancing) achieved an accuracy of 69.55% but had poor recall for the minority class.
  - The Logistic Regression model, after tuning, achieved an accuracy of 79.01%, showing better generalization.
  - Key performance metrics:
      - Random Forest: Precision = 0.49, Recall = 0.70, F1-score = 0.58.
      - Logistic Regression: Precision = 0.66, Recall = 0.61, F1-score = 0.63.
      - ROC-AUC Score (Logistic Regression): 0.6258.
      - PR-AUC Score (Logistic Regression): 0.4375.

### Key Insights
  - The Logistic Regression model, after hyperparameter tuning, outperformed the Random Forest classifier in accuracy and recall for the minority class.
  - Class imbalance significantly impacted initial model performance, and applying SMOTE alongside class balancing techniques improved recall.
  - Hyperparameter tuning played a crucial role in optimizing model performance, particularly for Logistic Regression.

### Future Improvements
  - One future development I could consider is enhancing feature engineering by creating new features derived from existing ones. For example, I could generate interaction features that might reveal more complex patterns in the data. Additionally, I could explore creating polynomial features to capture non-linear relationships or introduce domain-specific features that might be more informative.
  - Another improvement could involve advanced handling of class imbalance. While I applied SMOTE to balance the dataset, I could further investigate techniques like ADASYN or Borderline-SMOTE to generate synthetic samples more focused on the decision boundary.
  - I could also experiment with ensemble methods like Balanced Random Forest or EasyEnsemble, which are designed to work well with imbalanced data, or incorporate cost-sensitive learning to weigh misclassifications of the minority class more heavily
