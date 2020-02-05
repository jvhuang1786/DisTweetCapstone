# Disney Twitter Project Sentiment Analysis

## Capstone Project For Springboard on Sentiment Analysis between Tokyo Disney Resort and Anaheim Disney Resort

## Introduction

This is a sentiment analysis that compares Japanese and English tweets on their sentiment towards the Disney Parks.
Tweets were collected during the Halloween event Sept 24th, 2019 to Nov 1st, 2019 and during a portion of the Chinese New Year event Jan 30th to Feb 2nd.


All data was collected with the Twitter API collecting live stream tweets.

    Around 270000 tweets were collected after the cleaning process for Halloween
    Around 26000 tweets were collected after the cleaning process for Chinese New Year

**Goal is to see if Japanese sentiment towards Disney is more positive than English twitters users.  A model will be developed
to classify sentiment and user base and then LDA topic analysis will be done**

### Libraries/Modules

Need to install the following

<table>
<tr>
  <td>Data Wrangling/Cleaning</td>
  <td>Tweepy, re, emoji, numpy, bs4, pandas, document, string</td>
</tr>

<tr>
  <td>Exploratory Data Analysis</td>
  <td>string, seaborn, matplotlib, bokeh, plotly, spacy, scattertext, folium, basemap, wordcloud, counter </td>

</tr>

<tr>
  <td>Stats</td>
  <td>scipy.stats, nltk, textblob, asari</td>
</tr>

<tr>
  <td>Machine Learning</td>
  <td>sklearn, gensim, spacy, imblearn</td>

</table>


## Cleaning/Wrangling

Main steps of cleaning/wrangling the data:

     Getting the text out of nested dictionaries    
     Refilling the location with Geocoder    
     Keeping columns we are going to use for EDA    
     Text Preprocessing for ML model


* [Wrangling and Cleaning for EDA and Stats](https://github.com/jvhuang1786/DSCcareertrack/blob/master/capstone1/diswranglev11.ipynb)


* [Text Preprocessing](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/DISprepo.ipynb)


## Explanatory Data Analysis

Visualizations for Japanese and English Twitter users:

     WordCloud
     BaseMap
     Folium interactive map
     Hashtag, mention emoji count
     Popular tweet days
     Tweet peaks with plotly
     Word Count
     ScatterText

* [Exploratory Data Analysis](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/disEDA.ipynb)

* [Interactive Folium Map](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/twittermap.html)

* [ScatterText Interactive](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/disScatterText.ipynb)


## Stats

Statistical tests Vader and textblob to see Japanese vs English sentiment and Tokyo Disney vs Anaheim Disney:

       Mean
       Correlations
       Difference between means
       testing normalization, qq plot, shapiro normality test
       Mann-Whitney U Test

* [Statistical tests](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/disStats.ipynb)

## Machine Learning


### Supervised 

Ran a base model to see which document term matrix might work best.
    
    CountVectorizer
    TFDIFVectorizer
    Word2Vec
    
* [Document Term Matrix Evaluation](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/Dis_model_eval.ipynb)

Hypertuning using four algorithms 

    KNN
    Random Forest
    Multinomial Naive Bayes
    Logistic Regression 
    
* [Hyperparameter Tuning](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/dis_model_tune.ipynb)

Random Forest and Naive Bayes gave the best score.  Decided to see what was the best number of features, ngrams 
or min_df, max_df to use.

* [Random Forest](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/dis_rf_tuner.ipynb)

* [Random Forest (1,3) grams](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/dis_rf_tuner_ngrams.ipynb)

* [Naive Bayes](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/dis_Naive_bayes_tuner.ipynb)

* [Naive Bayes (1,3) grams](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/dis_naive_bayes_tuner_ngrams.ipynb)

### Unsupervised 

LDA topic modeling was done to see if we could generate topics after we separated the tweets into Japanese negative and positive,
English Negative and positive. 

* [Japanese Positive](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/dis_LDA_pos_ja.ipynb)

* [Japanese Negative](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/dis_LDA_neg_ja.ipynb)

* [English Positive](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/dis_LDA_pos_en.ipynb)

* [English Negative](https://nbviewer.jupyter.org/github/jvhuang1786/DisTweetCapstone/blob/master/dis_LDA_neg_en.ipynb)

## Model Using Chinese New Year Data

Model then was run on a completely new data set after pickling our best two models.  This page contains start to finish.
Cleaning the text, to a few EDAs, to using the model and topic modeling. 

* [Model Evaluation](https://github.com/jvhuang1786/DisTweetCapstone/blob/master/dis_model_production.ipynb) 

## Author

* Justin Huang

