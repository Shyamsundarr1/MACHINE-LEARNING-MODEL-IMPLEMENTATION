# MACHINE-LEARNING-MODEL-IMPLEMENTATION

*COMPANY*: CODTECH IT SOLOUTIONS

*NAME*: KEESARI SHYAMSUNDAR REDDY

*INTERN ID*: CT6WNYN

*DOMAIN*: PYTHON PROGRAMMING

*DURATION*: 6 WEEKS

*MENTOR*: NEELA SANTOSH

*PLATFORM USED*: JUPITER NOTEBOOK

*TOOLS USED*: VIRTUAL ENVIRONMENT

*DESCRIPTION*:

This project involves constructing a machine learning model to identify spam messages using Python's powerful libraries. Here's an in-depth look at each step involved:

Dataset Loading: The journey begins with loading your dataset, spam_dataset.csv, into a pandas DataFrame. This dataset comprises text messages and their corresponding labels (either spam or not spam). Pandas is a versatile library that simplifies data manipulation and analysis.

Text Preprocessing: Text data cannot be directly fed into a machine learning model. It needs to be converted into a numerical format. The CountVectorizer from the sklearn.feature_extraction.text module is utilized for this purpose. This vectorizer transforms the text data into a matrix of token counts, where each unique word is treated as a feature and its occurrence is counted. The resulting matrix, X, holds the numerical representation of the text data, while y contains the labels.

Dataset Splitting: To evaluate the model's performance, you split the dataset into training and testing sets. This is achieved using the train_test_split function from sklearn.model_selection. By allocating 80% of the data for training and 20% for testing, you ensure the model is trained on a large portion while reserving a portion for validation. Setting a random_state ensures reproducibility of the split.

Model Creation and Training: The heart of the project is the Multinomial Naive Bayes (MultinomialNB) model, a probabilistic classifier well-suited for text data. It leverages word frequency to predict the class of a message. The model.fit(X_train, y_train) method trains the model on the training data, enabling it to learn the patterns and relationships within the data.

Making Predictions: After training, the model is ready to predict labels for the unseen test data (X_test). This is done using the model.predict method, which generates the predicted labels (y_pred) for the test set. These predictions are then compared to the actual labels to evaluate the model's performance.

Evaluation Metrics: Assessing the model's performance involves calculating various metrics:

Accuracy: It measures the proportion of correctly classified messages out of the total. The accuracy_score function computes this metric, giving a quick overview of the model's effectiveness.

Confusion Matrix: This matrix provides a breakdown of true positives, true negatives, false positives, and false negatives. The confusion_matrix function helps visualize the types of errors made by the model, offering insights into its strengths and weaknesses.

Classification Report: This report details precision, recall, and F1-score for each class. Precision indicates the accuracy of the positive predictions, recall measures the ability to identify all positive instances, and the F1-score is the harmonic mean of precision and recall. The classification_report function generates this comprehensive report.

*LIBRARIES*:

pandas: For data manipulation and analysis, specifically for loading and handling the dataset.

sklearn.feature_extraction.text: Specifically the CountVectorizer class, to convert text data into a numeric format.

sklearn.model_selection: Specifically the train_test_split function, to split the dataset into training and testing sets.

sklearn.naive_bayes: Specifically the MultinomialNB class, to create and train the Naive Bayes classifier.

sklearn.metrics: Specifically the accuracy_score, confusion_matrix, and classification_report functions, to evaluate the model's performance.

These libraries provide the necessary tools to preprocess the text data, train the model, make predictions, and evaluate the results effectively.

#OUTPUT

![Image](https://github.com/user-attachments/assets/07b89105-f33f-4887-9885-2a35e6b295df)
