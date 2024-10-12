# Customer Churn Prediction

This project involves building a predictive model to identify customers likely to churn in a telecom company. The dataset contains various features such as account length, total call durations, international and voicemail plan statuses, customer service call counts, and more. The goal is to perform exploratory data analysis (EDA), build a churn prediction model, evaluate its performance, and deploy it to AWS EC2 using Docker with data stored in MongoDB.

## Project Overview

### 1. Exploratory Data Analysis (EDA)
The EDA will involve:
- Summarizing numerical features (e.g., `Total day minutes`, `Total night charge`).
- Understanding categorical variables (e.g., `International plan`, `State`).
- Investigating correlations between the features and the target variable `Churn`.
- Checking for class imbalance in the `Churn` feature.

### 2. Model Development
The dataset will be split into training and testing sets using a stratified split to maintain the proportion of churners and non-churners. The following models will be trained and evaluated:
- **Logistic Regression**: For a baseline and interpretability.
- **Random Forest**: To capture non-linear relationships and feature importance.
- **XGBoost**: For optimized performance on tabular data.

The models will be evaluated using:
- Accuracy
- Precision, Recall, and F1-Score (to handle imbalanced data)
- ROC-AUC
- Confusion matrix

### 3. Model Deployment
- **AWS EC2**: The trained model will be deployed on an AWS EC2 instance.
- **Docker & ECR**: The model will be containerized using Docker and pushed to AWS Elastic Container Registry (ECR).
- **MongoDB**: The customer data and prediction results will be stored in a MongoDB database for easy retrieval and storage.

## Libraries Used
The following libraries will be used in the project:
- **Python**: Core language for model development
- **Pandas**: For data manipulation and preprocessing
- **NumPy**: For numerical operations
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For building and evaluating machine learning models
- **XGBoost**: For training gradient boosting models
- **Flask**: For creating an API for the model to make predictions
- **Docker**: For containerizing the model
- **AWS SDK**: For interacting with AWS services
- **Pymongo**: For interacting with MongoDB

## Setup Instructions
1. Clone the repository:
    ```bash
    git clone https://github.com/yourusername/customer-churn-prediction.git
    cd customer-churn-prediction
    ```

2. Install the required Python libraries:
    ```bash
    pip install -r requirements.txt
    ```

3. Set up MongoDB:
    - Create a MongoDB cluster or use a local instance.
    - Update the connection string in the `.env` file with your MongoDB credentials.

4. Train the model:
    ```bash
    python train.py
    ```

5. Deploy the model:
    - **Docker**: Build the Docker image and push to AWS ECR.
    - **AWS EC2**: Launch an EC2 instance and pull the Docker image.
    - **MongoDB**: Ensure the MongoDB instance is accessible for data storage.

## Potential Challenges
- **Imbalanced Dataset**: Will handle using techniques like SMOTE or class weighting.
- **Data Drift**: The model will be monitored over time for performance degradation.
- **Interpretability**: Using SHAP values to explain complex models like XGBoost.

## Future Improvements
- Implementing a continuous integration/continuous deployment (CI/CD) pipeline.
- Adding more advanced hyperparameter tuning.
- Monitoring model performance with AWS CloudWatch.

## Contact
For any questions or issues, please feel free to reach out at your.email@example.com.
