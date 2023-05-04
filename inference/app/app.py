import json

import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from gensim import models
from gensim.corpora import Dictionary, MmCorpus

from flask import Flask, Response, request
import pandas as pd
from io import StringIO
import json
import traceback
from flask_cors import CORS, cross_origin

import os
import re
import numpy as np

from waitress import serve

import nltk
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *

from nltk.corpus import stopwords

import spacy


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'
#cors = CORS(app, resources={r"/predict": {"origins": "*"}})
id2word = Dictionary.load('dict')
lda = models.ldamodel.LdaModel.load('model')

nltk.download('wordnet')
nltk.download('stopwords')
nlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])

# establish list of common stop words
stop = stopwords.words('english')
custom_stopwords = ['b','ha', 'le', 'u', 'wa', 'b4', 'bd', 'sn', 'could', 'doe', 'might', 'must', 'need', 'sha', 'wo', 'would']
stop.extend(custom_stopwords)
stop = set(stop)

def nltk_stopwords():
    return set(nltk.corpus.stopwords.words('english'))
    
def sentence_to_words(sentence):
    return gensim.utils.simple_preprocess(str(sentence), deacc=True)

# Define function for stopwords, bigrams, trigrams and lemmatization
def remove_stopwords(text):
    return [word for word in simple_preprocess(str(text)) if word not in stop]

def make_bigrams(text, bigram_mod):
    return bigram_mod[text]

def make_trigrams(text, trigram_mod, bigram_mod):
    return trigram_mod[bigram_mod[text]] 

def lemmatization(text, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    """https://spacy.io/api/annotation"""
    text_out = []
    doc = nlp(" ".join(text)) 
    text_out.append([token.lemma_ for token in doc if token.pos_ in allowed_postags])
    return text_out
    


@app.route("/")
def hello():
    return "This is a Gensim-serving app"

@app.route("/ping", methods=['GET'])
@cross_origin()
def ping():
    """
    Determine if the container is healthy by running a sample through the algorithm.
    """
    try:
        return Response(response='{"status": "ok"}', status=200, mimetype='application/json')
    except:
        return Response(response='{"status": "error"}', status=500, mimetype='application/json')

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    """
    Do an inference on a single request.
    """

    try:
	    if request.content_type == 'application/json':
	    	request_data = request.get_json()
	    	text = request_data["text"]
	    	data = re.sub('\S*@\S*\s?', '', text)
	    	# Remove new line characters 
	    	data = re.sub('\s+', ' ', data)   
	    	# Remove distracting single quotes 
	    	data = re.sub("\'", "", data)
	    	data_words = sentence_to_words(data)
	    	# Build the bigram and trigram models
	    	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
	    	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
	    	# Faster way to get a sentence clubbed as a trigram/bigram
	    	bigram_mod = gensim.models.phrases.Phraser(bigram)
	    	trigram_mod = gensim.models.phrases.Phraser(trigram)
	    	
	    	# Remove Stop Words
	    	data_words_nostops = remove_stopwords(data_words)
	    	
	    	# Form Trigrams
	    	data_words_trigrams = make_trigrams(data_words_nostops, trigram_mod, bigram_mod)
	    	data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
	    	
	    	corpus = [id2word.doc2bow(text) for text in data_lemmatized]

	    	for i, row in enumerate(lda[corpus]):
	    	    row = sorted(row, key=lambda x: x[1], reverse=True)
	    	    for j, (topic_num, prop_topic) in enumerate(row):
	    	        if j == 0:
	    	            wp = lda.show_topic(topic_num)
	    	            topic_keywords = ", ".join([word for word, prop in wp])
	    	            request_data["dominant_topic"] = topic_num
	    	            request_data["perc_contribution"] = str(round(prop_topic,4))
	    	            request_data["topic_keywords"] = topic_keywords
	    	        else:
	    	            break

	    else:
	        return Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')
	    results_str = json.dumps(request_data)

	    # return
	    return Response(response=results_str, status=200, mimetype='application/json')
    except Exception:
    	return traceback.format_exc()

@app.route('/invocations', methods=['POST'])
@cross_origin()
def invocations():

    try:
	    if request.content_type == 'application/json':
	    	request_data = request.get_json()
	    	text = request_data["text"]
	    	data = re.sub('\S*@\S*\s?', '', text)
	    	# Remove new line characters 
	    	data = re.sub('\s+', ' ', data)   
	    	# Remove distracting single quotes 
	    	data = re.sub("\'", "", data)
	    	data_words = sentence_to_words(data)
	    	# Build the bigram and trigram models
	    	bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100) 
	    	trigram = gensim.models.Phrases(bigram[data_words], threshold=100)
	    	# Faster way to get a sentence clubbed as a trigram/bigram
	    	bigram_mod = gensim.models.phrases.Phraser(bigram)
	    	trigram_mod = gensim.models.phrases.Phraser(trigram)
	    	
	    	# Remove Stop Words
	    	data_words_nostops = remove_stopwords(data_words)
	    	
	    	# Form Trigrams
	    	data_words_trigrams = make_trigrams(data_words_nostops, trigram_mod, bigram_mod)
	    	data_lemmatized = lemmatization(data_words_trigrams, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])
	    	
	    	corpus = [id2word.doc2bow(text) for text in data_lemmatized]

	    	for i, row in enumerate(lda[corpus]):
	    	    row = sorted(row, key=lambda x: x[1], reverse=True)
	    	    for j, (topic_num, prop_topic) in enumerate(row):
	    	        if j == 0:
	    	            wp = lda.show_topic(topic_num, 30)
	    	            topic_keywords = ", ".join([word + ":" + str(prop) for word, prop in wp])
	    	            request_data["dominant_topic"] = topic_num
	    	            request_data["perc_contribution"] = str(round(prop_topic,4))
	    	            request_data["topic_keywords"] = topic_keywords
	    	        else:
	    	            break

	    else:
	        return Response(response='This predictor only supports Json data', status=415, mimetype='text/plain')
	    results_str = json.dumps(request_data)

	    # return
	    return Response(response=results_str, status=200, mimetype='application/json')
    except Exception:
    	return traceback.format_exc()


if __name__ == "__main__":

    # Only for debugging while developing
    # app.run(host='0.0.0.0', debug=False, port=8000)
    # To be used for production (Waitress)
    serve(app, host='0.0.0.0', port=8080)