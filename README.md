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


## Project Overview
The **WiDs Datathon Global Challenge** Kaggle competition is an opportunity for BreakThrough Tech AI student fellows to apply and enhance their data science skills while working on an impactful and important problem. The BreakThrough Tech AI Program provides students with a strong foundation in machine learning, equipping them with the skills needed to tackle real-world challenges like the one presented in this competition.
### Goal of the Challenge 
The goal of his challenge is to build a model using **fMRI data** and **Socio-demographic information** to predict:
1. Individual's sex (Male or Female)
2. Individual's ADHD diagnosis
By analyzing the data, participants aim to uncover patterns and insights that contribute to a better understanding of ADHD and reveal potential biases in machine learning models used in this field.

### Real World Significance and Impact
ADHD is one of the most common neurodevelopmental disorders, affecting approximately **11% of adolescents**. However, its manifestation varies significantly between **males and females**, with females often being underdiagnosed due to differences in symptom presentation. Undiagnosed ADHD can lead to long-term mental health struggles, making it difficult for individuals to access the resources and support they need to thrive.

## Data Exploration 
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


