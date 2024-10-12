# Customer Churn Prediction

This project involves building a predictive model to identify customers likely to churn for a telecom company. The dataset includes features like account length, call durations, international and voicemail plan statuses, customer service call counts, and more. The goal is to perform exploratory data analysis (EDA), build a churn prediction model, evaluate its performance, and deploy it to AWS EC2 using Docker, with data stored in MongoDB.

## Project Overview

### Steps Covered in the Project

1. **Problem Definition**
   - Understanding the business problem of customer churn and how machine learning can provide a predictive solution.

2. **Data Collection**
   - Using the `Data_Science_Challenge.csv` dataset, which contains customer data from a telecom company.

3. **Data Cleaning and Preprocessing**
   - Handling missing values, correcting data types, and addressing data inconsistencies to ensure data quality before model building.

4. **Exploratory Data Analysis (EDA)**
   - Visualizing and statistically analyzing features to uncover important patterns, relationships, and trends that influence customer churn.

5. **Feature Engineering**
   - Transforming raw data into meaningful features, including scaling numerical data, encoding categorical variables, and selecting key features that impact churn prediction.

6. **Model Selection**
   - Trying a variety of machine learning models, including Logistic Regression, Random Forests, and XGBoost, to determine which algorithm provides the best predictive performance.

7. **Model Training**
   - Training selected models using cross-validation techniques to validate their performance.

8. **Model Evaluation**
   - Evaluating models using metrics like:
     - Accuracy
     - Precision, Recall, and F1-Score (to handle imbalanced classes)
     - ROC-AUC for overall model performance
     - Confusion Matrix to assess predictions

9. **Hyperparameter Tuning**
   - Optimizing model performance by adjusting hyperparameters using techniques like GridSearchCV or RandomizedSearchCV.

10. **Pipeline Creation for Automation**
    - Automating data preprocessing and model training with a machine learning pipeline to improve efficiency and scalability.

11. **Model Deployment**
    - **AWS EC2**: Deploying the model on an AWS EC2 instance.
    - **Docker & AWS ECR**: Containerizing the model using Docker and pushing it to AWS Elastic Container Registry (ECR) for deployment.
    - **MongoDB**: Storing customer data and prediction results in a MongoDB database.

12. **Making Predictions on New Data**
    - Testing the deployed model by feeding it new data and generating predictions on customer churn in real-time.

13. **Monitoring and Maintenance**
    - Monitoring the deployed model’s performance over time to detect data drift and retrain as necessary to maintain prediction accuracy.

### Libraries Used
- **Python**: Core programming language for development
- **Pandas**: For data manipulation and preprocessing
- **NumPy**: For numerical operations
- **Matplotlib & Seaborn**: For data visualization
- **Scikit-learn**: For building and evaluating machine learning models
- **XGBoost**: For advanced gradient boosting models
- **Flask**: To build an API for serving the model and making real-time predictions
- **Docker**: To containerize the model for deployment
- **AWS SDK**: For interacting with AWS services (ECR, EC2)
- **Pymongo**: For connecting and interacting with MongoDB

### Setup Instructions

1. Clone the repository:
    ```bash
    git clone https://github.com/natek-1/churn_prediction.git
    cd churn_prediction
    ```

2. Install the required Python libraries:
    ```bash
    conda create -y python=3.9 -n churn
    conda activate churn
    pip install -r requirements.txt
    ```

3. Set up MongoDB:
    - Create a MongoDB cluster (or use a local instance).
    - Update the MongoDB connection string in the `.env` file with your credentials.

4. Train the model:
    ```bash
    python train.py
    ```

5. Deploy the model:
    - **Docker**: Build the Docker image for the model and push it to AWS ECR.
    - **AWS EC2**: Launch an EC2 instance, pull the Docker image, and run the model.
    - **MongoDB**: Ensure the MongoDB instance is accessible for storing predictions and customer data.

### Potential Challenges

- **Imbalanced Dataset**: The churn dataset is likely to be imbalanced, so we will employ techniques like SMOTE, undersampling, or class weighting to handle this issue.
- **Data Drift**: Customer behavior might change over time, so it will be important to monitor the model and periodically retrain it to maintain its predictive power.
- **Model Interpretability**: For complex models like XGBoost, we’ll use SHAP values to explain individual predictions.

### Future Improvements
- Implement a continuous integration/continuous deployment (CI/CD) pipeline for seamless updates to the model.
- Explore more advanced feature engineering techniques for better performance.
- Use AWS CloudWatch for real-time monitoring of the deployed model's performance.
