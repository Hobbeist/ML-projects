# Functions to tokenise text for sentiment analysis
import nltk 
import re
import string
from gensim.models import Phrases
from gensim.models.phrases import Phraser

def createToken(input, stopWords, urduNames):

  '''Take the text of a review and create tokens
  
  Parameters
    ---------------------------------------
  input: text/comment/post column of a pandas data frame
  
  stopWords: A list of stopwords
  
  urduNames: A list of common Urdu names
  '''
  
  # First of all, given we have identified that there are some comments that are one word long, but are 
  # actually a sentnce with dots instead of whitespace, we replace the dots with whitspace for those
  if len(input.split()) == 1:
    tokenisedWords = input.replace('.', ' ')
    tokenisedWords = [word.lower() for word in nltk.word_tokenize(tokenisedWords) if len(word) > 1]
    # Remove numbers
    tokenisedWords = [re.sub(r'[0-9 \n\.]','',word) for word in tokenisedWords]
    # Remove puntuations
    tokenisedWords = [re.sub(r'['+string.punctuation+']', '',word) for word in tokenisedWords]
    # Remove stop words and common names
    tokenisedWords = [word for word in tokenisedWords if not word in stopWords and word not in urduNames]
  else:
    # Split comment into words; 
    # apply only to comments that contain more than one word; 
    # return the word in lower case. 
    tokenisedWords = [word.lower() for word in nltk.word_tokenize(input) if len(word) > 1]
    # Remove numbers
    tokenisedWords = [re.sub(r'[0-9 \n\.]','',word) for word in tokenisedWords]
    # Remove puntuations
    tokenisedWords = [re.sub(r'['+string.punctuation+']', '',word) for word in tokenisedWords]
    # Remove stop words and common names
    tokenisedWords = [word for word in tokenisedWords if not word in stopWords and word not in urduNames]
  return tokenisedWords





def tokeniseAll(posts, stopWords, urduNames):
    
    '''Function to tokenise all comments in the file, including ngrams
    
    Parameters
    ---------------------------------------
    comments: the pandas data frame column containing the comments, transformed into a list
    
    stopWords: A list of stopwords
    
    urduNames: A list of common Urdu names'''
    
    #posts = comments.tolist()
    n_grams = 3
    tokenized_corp=[]
    for doc in posts:
        tokenized_corp.append(createToken(doc, stopWords, urduNames))
    
    # Add n_grams
    bigram = Phrases(tokenized_corp, min_count=5, threshold=10)
    trigram = Phrases(bigram[tokenized_corp], threshold=10)
    bigram_mod = Phraser(bigram)
    trigram_mod = Phraser(trigram)
  
    if n_grams > 1:
        for i, doc in enumerate(tokenized_corp):
            tokenized_corp[i] = bigram_mod[doc]
            if n_grams > 2: 
                tokenized_corp[i] = trigram_mod[bigram_mod[doc]]
    return tokenized_corp