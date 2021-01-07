## Exploring available data and collaborating with a cross-functional team

This month I worked with a team of developers to improve a web application which aims to highlight incidents of police use of force within the United States. These instances were then used to create a visualization on a map to show where these instances have occured. This project was proposed by Human Rights First, a US-Based human rights organization. HRF strives to influence the US Government and private companies to respect human rights and the rule of law, and seek accountability and reform where they fail. The goal of this app is to make the data easily accessible to journalists and other interested parties who are able to document and spread awareness about these incidents. 

Link to the DS application can be found here: [DS API](https://b-ds.humanrightsfirst.dev/)

Link to our team's Web application here: [Web Application](https://www.b.humanrightsfirst.dev/)

My initial questions about this project were about the data available and how it is being used. Currently, there are other organizations who have developed very similar web applications. For example, [2020PoliceBrutality](https://github.com/2020PB/police-brutality) started as a subreddit where people could report incidents of police brutality, and now they collect reports in a github repository. Multiple sites have been built using the 2020PB dataset. Another group is [8:46 Police Brutality](https://github.com/949mac/846-backend/). 846 has collected their data into an API which is regularly updated. These organizations are doing a great job of collecting incidents of police use of force. However, our Stakeholder informed us that there were still incidents they had noticed that were not being reported, and they would like to pull in from more data sources in the future. In addition to bringing in more data sources, it was also important to reduce the number of duplicate incidents. 

## Making decisions about the data

As a team we spent a lot of time brainstorming how to pull in data from additional sources and reduce the number of duplicated incidents. This is a very challenging problem. One idea was to look for duplicates based on time and location, however not all data sources have precise coordinates for where the event took place. Additionally, it is possible for more than one event to happen at the same time, especially where there are protests and a large police presence. Another idea was to compare link URLs in order to remove duplicates. Depending on the source, most of these posts also included a link, but it is also possible for multiple links to describe the same event. 

Our first decision was how to incorporate the data from 2020PB and the 846 API. I suspected that because these groups appeared to be affiliated the data that they used would be very similar. My first task was to compare these two sources. In order to connect to the 2020PB subreddit, I used [PRAW](https://praw.readthedocs.io/en/latest/): The Python Reddit API Wrapper. This enabled me to pull the newest posts from r/2020pb and create a dataset containing 976 incidents. Initially I thought that one benefit to using the data directly from the subreddit would be including the body of the post. This would provide more text data that could be useful for training an NLP text classification model. However, I found that very few of the posts in this subreddit included text within the body of the post, and that most of the text data is in the title and the tags. 

Next, I connected to the 846PoliceBrutality API, which contained over 1200 incidents. The API also contained title and tag information which was comparable to the same information found on 2020pb. Additionally, it contained more detailed location data, including geolocation coordinates, City and State. Some other benefits to using 846 is that the data is regularly updated and all observations are confirmed incidents of police use of force. The moderators also provide submission guidelines and have made an effort to prevent duplication. Since all data from the 846 API contains relevant incidents, we decided to connect the API directly to our app. 

Another data source I investigated was Twitter. The benefit of using Twitter is that users post events in real time which would make it possible to catch new incidents. One of the downsides of using Twitter, however, is the large amount of data that needs to be filtered before finding relevant incidents. At one point I thought it would be a good idea to collect the usernames of the largest police departments in the US, since many of them tweet current updates. However, while collecting relevant users would have cut down on the amount of data that needs to be collected, this system would have been one sided. A relevant tweet about police presence could come from any user. 

In order to investigate tweets I used the [Tweepy](http://docs.tweepy.org/en/latest/index.html) python library to connect to the Twitter API and collect tweets into an sqlite database. I used the StreamListener method to create a stream and filter for tweets containing the word 'police'. The stream filter could be further customized to include hashtags and user IDs. 

Example of StreamListener using Tweepy:
~~~
class StreamListener(tweepy.StreamListener):
  def on_status(self, status):
  """collecting original tweets, not retweets"""
      if not 'RT @' in status.text:
          text = status.text
          coords = status.coordinates
          source = status.user.url
          created = status.created_at
      
      if coords is not None:
          coords = json.dumps(cords)
      """insert into table"""
      table = db['tweets']
      try:
          table.insert(dict(
              text = text,
              coordinates = coords,
              source = source,
              created = created,
              ))  
~~~
The Twitter API and the 846 API contain similar features, such as geocoding and links URLs. This means relevant tweets could be compared to the 846 API, and if needed, inserted into the CSV or database. While exploring the Twitter data I was able to use the 846 API to create a list of links, and then check my Twitter dataset for duplicates. 

![](https://raw.githubusercontent.com/cdixson-ds/cdixson-ds.github.io/master/img/Twitter_concept.png)
    
Another option for bringing in additional data that I considered was [NewsAPI](https://newsapi.org/). NewsAPI returns news articles from multiple sources which can also be filtered by keyword and location. One of the benefits to using NewsAPI is that it contains more text data within the title, description and content which can be used to train an NLP classification model. One of the disadvantages is that location data is not easily accessible, and would need to be found within the article's content. 

## Training a model

Everyone on the data science team for this project proposed different classification models. One idea we all had in common was to use a Nearest neighbors algorithm. Nearest neighbors is an unsupervised learning algorithm which is able to classify data based on proximity or similarity. When experimenting with the model I used  In order to train a Nearest neighbors model, our training data would need to include posts or keywords from posts describing police use of force. Our model can then be used to find new instances of police use of force from other datasets. My idea was to use tokenized text data from the 846 dataset to train a classification model which could be used to find similar posts from other sources. 

In order to tokenize the training data, I used [spaCy](https://spacy.io/) which is a Natural Language Processing package in Python. With spaCy I used a function which would remove stop words, punctuation, and pronouns from the 846 text data. Additionally, the tokens would be lemmatized, or grouped together based on inflection. 

Example of how to use tokenization with spaCy:
~~~
import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_sm")

"""Instantiating tokenizer"""
tokenizer = Tokenizer(nlp.vocab)

def tokenize(doc):
    lemmas = []
    doc = nlp(doc)
    for token in doc: 
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_ != 'PRON'):
            lemmas.append(token.lemma_)
    return lemmas
~~~

![](https://raw.githubusercontent.com/cdixson-ds/cdixson-ds.github.io/master/img/token_data.PNG)

The plot above shows the top 20 tokens tokens found in the 846 dataset excluding 'police'. The next step was to use TfidfVectorizer, which is in the [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) feature extraction library. TF-IDF, which stands for term frequency-inverse document frequency, is a statistical measure that evaluates how relevant a word is to a document within a collection of documents. In this case, a numeric value is assigned based on how many times it appears within a post, and is offset by how many times it appears in the other posts. These values were then used to train a Nearest neighbors model.  

## Current product

Our data science team was able to deploy the [Data Science API](https://b-ds.humanrightsfirst.dev/) using FastAPI, Python, Docker, and AWS Elastic Beanstalk. The API is used to supply data to the backend of the main [Web Application](https://www.b.humanrightsfirst.dev/). The DS app is currently connected to the 846 API. Users are able to retrieve a JSON object, or list of JSON objects, based on report id, City, State, and State, or the entire dataset. The user is also able to reload the data, which will add any new instances from the 846 API. Finally, the app is also able to make a prediction based on text input. The predictive model that our team's app is currently using is a k-Nearest neighbors algorithm. This is similar to a Nearest neighbors algorithm, except labels have been assigned to the training data. The prediction is ranked by the severity of the police encounter from Rank 0, no police presence, to Rank 5 which is lethal force.  

## Future features

One idea that our team agreed would make a good feature, but we did not have time to implement, was an admin dashboard where new incidents can be approved based on the recommendations of a predictive model. This would be a great way to find new incidents of police use of force while reducing the number of false positives which are shown on the public app. It would be very difficult to determine if an event is a duplicate based only on its time and location or the link's URL. Additionally, it is very difficult to create a classification model that is accurate enough to not require some human oversight. 

## Takeaways 

This project has been a great opportunity to work in a team environment. One thing I learned about myself is that I really enjoy data exploration and asking questions about the integrity of a given dataset or how it can be put to use. This is an area that I naturally gravitated towards. I also had some great practice communicating with our group and practicing good git workflow. Overall, I believe we spent an appropriate amount of time in the planning phase and were able to deliver some great new features, such as connecting to a live API to deliver more up-to-date data and improving the classification model. 


