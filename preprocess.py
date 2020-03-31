import pandas as pd
import numpy as np
import re
import string

# A list of contractions from 
# http://stackoverflow.com/questions/19790188/expanding-english-language-contractions-in-python
contractions = { 
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are",
    "thx"   : "thanks"
}

def update_keywords(text):
    text = re.sub(r'wild fires', 'wildfire', text)
    text = re.sub(r'forest fires', 'forest fire', text)
    text = re.sub(r'body bags', 'body bag', text)
    text = re.sub(r'buildings burning', 'burning buildings', text)
    return text

def remove_emoji(text):
    emoji_pattern = re.compile("["
                           u"\U0001F600-\U0001F64F"  # emoticons
                           u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                           u"\U0001F680-\U0001F6FF"  # transport & map symbols
                           u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           u"\U00002702-\U000027B0"
                           u"\U000024C2-\U0001F251"
                           "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', text)

def remove_url(text):
    return re.sub(r"http\S+", "", text)

def remove_non_ascii(text):
    return ''.join([x for x in text if x in string.printable])

def remove_contractions(text):
    return contractions[text.lower()] if text.lower() in contractions.keys() else text

def remove_punctuations(text):
    text = str(text)
    punctuations = '''!()-[]{};:|'"\,<>./?@#$%^&*_~'''
    for x in text:
        if x in punctuations:
            text = text.replace(x, '')
    return text

def preprocess(df, preprocess_keywords=False):
    if preprocess_keywords:
        # preprocess keywords
        df['keyword'].fillna("none", inplace = True)
        df['keyword'] = df['keyword'].apply(lambda k: update_keywords(k))
        df['keyword'] = df['keyword'].apply(lambda k: k.lower())

    # preprocess text
    df['text'] = df['text'].apply(lambda k: remove_url(k))
    df['text'] = df['text'].apply(lambda k: remove_non_ascii(k))
    df['text'] = df['text'].apply(lambda k: remove_emoji(k))
    df['text'] = df['text'].apply(lambda k: remove_contractions(k))
    df['text'] = df['text'].apply(lambda k: remove_punctuations(k))
    df['text'] = df['text'].apply(lambda k: k.lower())
 
    return df
