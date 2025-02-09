# Sentiment Analysis with Data Preprocessing and Model Training

In this project, I took a dataset of movie reviews and trained a model to predict whether a review is positive or negative. 
I used the IMDB dataset, which contains 50,000 reviews. 
The dataset is balanced, with 25,000 reviews labeled as positive and 25,000 as negative.
We started with data preparation, then went through the preprocessing steps, and finally trained and evaluated some models to find the best fit. 
For interpretability reasons, we chose Logistic Regression as the best model and saved it for later use.
We also saved the model for later use.


Here is a brief overview of the project:


## Data Preparation
We used the IMDB dataset, which contains 50,000 reviews. The dataset was balanced, with 25,000 reviews labeled as positive and 25,000 as negative. We did not have any missing values, but we did have some duplicates, which we kept.


## Data Preprocessing
We converted the labels to numerical values and preprocessed the text by converting it to lowercase, removing special characters, removing stopwords, and lemmatizing the text. We also did some n-gram analysis to understand the context of the reviews.


## Model Training to find best Model
We used TF-IDF to normalize the data and vectorize it. We split the data into training and testing sets and trained several classifiers to find the best one. We then used cross-validation to evaluate the models and selected Logistic Regression and Naive Bayes for further tuning.


## Hyperparameter Tuning
We used GridSearchCV to find the best parameters for Logistic Regression and Naive Bayes.
Even after hyperparameter tuning, both models performed similarly, with Logistic Regression having a slight edge.
To find the best model for sentiment analysis, we evaluated both models on the test set.

## Model Evaluation
We evaluated the models using accuracy, confusion matrix, and classification report. Both models performed well, with Logistic Regression having a slight edge in the number of correct predictions. Since both models are very similar, the final choice depends on the use case. The regression model is more interpretable, while the Naive Bayes model is faster to train and predict.

In most cases, a good interpretable model is preferred, since we could compute shape values and explain the model to stakeholders.


## Saving the Model
We saved the regression model as a pickle file for later use. This would be especially useful if we wanted to deploy the model in a production environment and the model training took a long time.