
<div align="center">

# Sentiment Analysis of Social Media Posts in Roman Urdu  
### Special Focus: Prediction of negative Sentiment  
<br></br>  

##### _Prepared by: Sebastian Rauschert, PhD_  
</div>

***
# Summary of this Data Science Project Folder  
This project folder contains everything necessary to reproduce the sentiment analysis on Roman Urdu social media posts.
The structure of the project is the following:

```
+-- data  
|   +-- 100-common-names.txt  
|   +-- Roman_Urdu_DataSet.csv  
|   +-- stopwords.txt  
+-- notebooks  
|   +-- 01-Sentiment-Analysis.ipynb  
+-- urdu_sentiment  
|   +-- __init__.py    
|   +-- plotting.py  
|   +-- tokenise.py  
+-- README.md  
+-- requirements.txt  
+-- setup.py  
```

First, `pip install requirements.txt` in a terminal within this project folder. This will install the dependencies.
Further, run `pip install .` in the top level of this folder to make sure that the `urdu_sentiment` module is loaded.

Alternatively, just inspect the jupyter notebook in the notebook folder to read up about the tasks performed.


# Table of Contents
1. [Background](#background)  
1.1 [Task](#task)   
1.2 [Urdu](#urdu)   
2. [Natural Language Processing](#natural_language_processing)  
2.1 [Common pre-processing for NLP](#common_pre_processing_for_nlp)  
2.2 [Machine Learning](#machinelearning)  
3. [Preprocessing](#preprocessing)  
3.1 [Getting the data](#getting_the_data)  
3.2 [Descriptives](#descriptives)  
3.3 [Comments: irregularities](#comments_irregularities)  
3.3 [Tokenisation](#tokenisation)  
4. [Modelling](#modelling)  
5. [Summary](#summary)  

***

# Background
## Task
I have not yet dabbled in natural language processing and sentiment prediction in machine leaarning, so I decided to give it a go.
I used the [Roman Urdu social media post data set](https://archive.ics.uci.edu/ml/datasets/Roman+Urdu+Data+Set) as an example (because: why make it easy?).



## Urdu
The data set, contains social media posts in _Roman Urdu_. As per [Wikipedia](https://simple.wikipedia.org/wiki/Urdu), Urdu is the national language of Pakistan. It is further spoken in some parts of India and sounds similar or same as Hindi. Urdu is usually written in the Persio-Arabic alphabet.  
[Roman Urdu](https://en.wikipedia.org/wiki/Roman_Urdu) on the other hand, is the name of the Urdu language written with the Latin or Roman script. According to Wikipedia, "despite this opposition of Roman Urdu by traditional Arabic script lovers, it is still used by most on the internet and computers due to limitations of most technologies as they do not have the Urdu script."


# Natural Language Processing
I only had limited knowledge on Natural Language Processing (NLP) before this project. Hence, I started with some research on the topic. This included how to clean text / natural language data, what common techniques are used to perform sentiment analysis and what the relevant python packages are.  

## Common pre-processing for NLP
To get an overview, I found the following links to be helpful, amongst others:  
-  [Preprocess text data](https://towardsdatascience.com/text-preprocessing-for-data-scientist-3d2419c8199d)  
-  [Naive Bayes Algorithms for sentiment analysis](https://towardsdatascience.com/sentiment-analysis-introduction-to-naive-bayes-algorithm-96831d77ac91)  
-  [n-grams of words](https://towardsdatascience.com/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing-9d9eef0fa058)  


### **Tokens** 

In NLP it is quite standard to break up a text into sentences or words. The derived pieces (sentence or word) are referred to as tokens.

<br></br>


### **Stop Words**

**Stop words** are common words that do not contribute much of the information in a text document. In english, such words would be for example ‘the’, ‘is’ and ‘a’. These words only add noise to the data, so it seems to be common practice to remove them. 
As I am not dealing with english, this is a bit more tricky. So first of all, I was searching for a list of Urdu stop words, which I found in [this git repo](https://raw.githubusercontent.com/haseebelahi/roman-urdu-stopwords/master/stopwords.txt).

<br></br>


### **(Unhelpful) nouns or names**

In order to clean the text further, it is useful to remove names and if possible, common nouns, that are unhelpful for the sentiment. I found this github repository, that contains all sorts of information about Roman Urdu, including [this list of very common names](https://raw.githubusercontent.com/Smat26/Roman-Urdu-Dataset/master/Urdu-Names/100-common-names.txt).

<br></br>

### **n-grams**  
This concept refers to the probability of n co-occuring words in a text. This can be useful for text completion, but also in sentiment analysis, where combinations of words might be useful in addition to single word tokens.

<br></br>

### **TF-IDF**

**TF-IDF**, short for 'Term Frequency - Inverse Document Frequency, creates a matrix, called corpus, of the total number of tokens (words) in the text, by the number of comments (tokens $\times$ \(n_{comments}\) ), where the cells are the token frequency in the document (number of occurences), expressed as a probability, weighted by the number of occurences of that token/word in the comment, as compared to the whole number of comments.
Simpler: The TF-IDF weights comment specific but overall rare words higher.  
As per [scikit library](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html#sklearn.feature_extraction.text.TfidfTransformer): 

> The goal of using tf-idf instead of the raw frequencies of occurrence of a token in a given document is to scale down the impact of tokens that occur very frequently in a given corpus and that are hence empirically less informative than features that occur in a small fraction of the training corpus.

***

## Machine Learning  

I have identified three approaches to perform sentiment analysis. 

- **(Multinomial) Logistic Regression**  
- **Naive Bayes**   
- **LSTM Neural Nets**  

There are several other possibilities, such as Support Vector Machine, and using pre-trained Neural Nets such as BERT and RoBERTa as well. Since the scope of this project was not to build the state of the art best performing model, but to get a hang of NLP and to set up a reproducible python data science project, I decided to focus on Multinomial Logistic Regression and Naive Bayes.  


# Preprocessing
## Getting the data
I downloaded the data set and the identified list of Roman Urdu stop words and common names, as my understanding is that it is immensly helpful to only retain the text that is actually meaningful in retaining a sentiment.

I have placed all files in the `/data` folder int this project.

## Descriptives
As always, my first step was to descriptively and visually inspect the data.
I looked at shape, first rows, class distributions and mislabelling of classes. Further, I removed rows with missing values.

## Comments: irregularities
Next, I inspected the comments to see if there were any irregularities. Are ther any that contain no words, or only one word?
Are there comments with irregularities (such as the identified dot instead of spaces in betwen words for some comments)? 
Are there any duplicates?

## Tokenisation
I tokenised the comments into words and 2 and 3-grams. Further, I removed punctuations, stop words and common nouns, as well as set all words to lower case. After that, I checked again if some of the comments were now without words, in case the tokenisation removed comments that only consisted of symbols.

**Important note** I do realise that I removed emojis in this step and there are many arguments against this, such as the fact that we communicate emotions often with emojis. If the aim of this task was to create a best in class model, I would investigate further how to handle emojis.

*** 

# Modelling
As stated above, I have used Multinomial Logistic Regression, as it is probably one of the simplest approaches for this task, and Naive Bayes, as this is commonly used in text classification tasks.  
My approach is to start from low complexity and work up to high complexity models, as it is not always the biggest hammer that is required to put the nail in.

<br></br>

I thought of how to proceed and briefly though of only modelling with the positive and negative comments, but quickly decided to not go ahead with this. After all, the real world will contain neutral comments as well.   
So first of all, I modelled the data with all three classes. After that, I combined the positive and neutral comments, to see wether this helped in the classification problem.

 ***
 
# Summary

There are many other ways of performing this analysis, and many likely better models. One method I have identified is deep learning based and implements a [Long Short Term Memory (LSTM) deep learning model](https://towardsdatascience.com/sentiment-analysis-using-lstm-step-by-step-50d074f09948).
Further, one can leverage the pre-trained [RoBERTa model](https://towardsdatascience.com/discover-the-sentiment-of-reddit-subgroup-using-roberta-model-10ab9a8271b8) to perform the task. Both are very likely to perform better on the sentiment analysis.  

With this project, however, I just wanted to polish up my python skills and to get some understanding on NLP and sentiment analysis more so than training the perfect classifier in a field that was new to me. Hence, I put emphasis on how to set up a data science project folder and how to make it minimally reproducible with the inclusion of a `requirements.txt`. 

Further, I created a local minimal python package with some basic functions that I used in the notebook. I could have included those in the notebook, as this is a small test project, but in a real world data science project with multiple collaborators, it would be handy to have the long code chunks placed separately.
