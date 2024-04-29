# Multi-Algorithm Machine Learning Project for Dry Bean Classification

This project aims to classify dry beans using three different machine learning algorithms: Random Forest, Support Vector Classifier (SVC), and Long Short-Term Memory (LSTM) neural network. The dataset used in this project is the Dry Bean Dataset obtained from the UCI Machine Learning Repository.

## Random Forest Classifier
- **Preprocessing**: The dataset is preprocessed and split into training and testing sets. Feature scaling and selection are performed using pipelines.
- **Model Tuning**: Grid search with cross-validation is used to tune the hyperparameters of the Random Forest classifier.
- **Model Evaluation**: The best model is evaluated on the test set using confusion matrix and classification report. Learning curves are plotted to visualize the model's performance.

## Support Vector Classifier (SVC)
- **Preprocessing**: Features are standardized, and the dataset is split into training and testing sets.
- **Hyperparameter Tuning**: Grid search with cross-validation is employed to find the optimal hyperparameters for the SVC model.
- **Model Evaluation**: The best model is evaluated on the test set using confusion matrix and classification report. Accuracy scores are calculated to assess model performance.

## Long Short-Term Memory (LSTM) Neural Network
- **Preprocessing**: The dataset is split into training and testing sets, and class labels are encoded.
- **Model Training**: A sequential LSTM model is built and trained using the training data.
- **Model Evaluation**: The trained model is evaluated on the test set using accuracy metrics and classification report. Confusion matrix is also generated to visualize the model's performance.

## Usage
1. **Data Loading**: The Dry Bean Dataset is fetched from the UCI repository using the `fetch_ucirepo` function.
2. **Preprocessing**: Data preprocessing, including feature scaling and selection, is performed for each algorithm.
3. **Model Training**: Train the Random Forest, SVC, and LSTM models using the preprocessed data.
4. **Evaluation**: Evaluate the trained models on the test set using appropriate evaluation metrics.
5. **Visualization**: Plot confusion matrices and learning curves to visualize model performance.

## Dependencies
- pandas: Data manipulation library for handling datasets.
- numpy: Numerical computing library for array manipulation.
- scikit-learn: Machine learning library for model training, evaluation, and preprocessing.
- matplotlib: Plotting library for data visualization.
- seaborn: Statistical data visualization library for creating informative and attractive visualizations.
- tensorflow: Deep learning framework for building and training neural networks.

## Note
- Ensure that the necessary dependencies are installed in your environment before running the script.
- Experiment with different hyperparameters and preprocessing techniques to optimize model performance.
- This project provides a comprehensive comparison of three different machine learning algorithms for dry bean classification.

## Contact
For any inquiries or support, please contact [arman13m99@gmail.com].
