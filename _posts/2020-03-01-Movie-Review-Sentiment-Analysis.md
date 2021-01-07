## For this project I deployed a web based app using a logistic regression model to predict the sentiment of a movie review

My app was deployed using [Plotly](https://plotly.com/dash/) and [Heroku](https://www.heroku.com/home) and can be found here: [Movie Review Sentiment Analysis](https://sentiment-movies-reviews.herokuapp.com/)

I had a lot of fun with this project. This is where I was able to research and implement some Natural Language Processing techniques for the first time. I used the bag-of-words model where text is represented as a bag(or multiset) of words without taking grammar or word order into account. 

![Word Cloud generated using training data](https://sentiment-movies-reviews.herokuapp.com/assets/wordcloud.png)

The dataset used to train the model came from IMDb, which contained 40,000 observations labeled either positive or negative. The distribution between both of these classes was fairly balanced, as both positive and negative classes were at roughly 50%. The average movie review was roughly 1,300 words. 

My first step was to clean the text by removing punctuation, symbols, HTML tags, and English stop words. Stop words are common words which are found frequently in a language, such as 'a', 'the', 'is', or 'at'. These words not only occur frequently, but are also not helpful to the model because they have very little meaning when compared to other lexical items. I used the [NLTK library](https://www.nltk.org/) to remove stop words.

My next step was to use [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html), part of the scikit learn library, to convert the reviews into a matrix of token counts, as well as [TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) in order to give each token a weighted value. Tf-idf stands for term frequency-inverse document frequency. The inverse document frequency is used to reduce the weight of words which occur very frequently, and increase the weight of terms which are more rare.

I tested multiple models while working on this project, however the app uses a logistic regression model to make it's predictions. Logistic regression is a statistical model that uses a logistic function used to predict a dependent, categorical variable. The model's training accuracy is 93%, and it's test accuracy rate is 89%. The confusion matrix for the test set is displayed below. The true positive and true negative values can be seen in yellow on the diagonal. These are the model's correct predictions. The number of false positive and false negative predictions are seen in purple.

![](https://sentiment-movies-reviews.herokuapp.com/assets/confusion_matrix.png)

The ROC AUC curve measures how well a classification model ranks predicted probabilities, and ranges from 0 to 1. It shows how much the model is able to distinguish between classes. This model's ROC AUC score is 0.96, indicating that it has a high measure of separability.

![](https://sentiment-movies-reviews.herokuapp.com/assets/roc_curve.png)

This was a valuable learning experience because it was my first time using NLP techniques to do sentiment analysis, and it was also the first time that I was able to build and deploy a web application. These are all skills that I am excited to learn and build on in the future.
