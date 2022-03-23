import pandas as pd
from string import punctuation
from nltk.tokenize import word_tokenize
from collections import Counter
import operator
import numpy as np
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import warnings
import string,re
warnings.filterwarnings('ignore')

def clean_doc(text):
	text = text.replace("[^a-zA-Z#]", " ")
	regex = re.compile('[%s]' % re.escape(string.punctuation))
	text = regex.sub('', text)
	text =  ' '.join([w.lower() for w in text.split() if len(w)>2])
	return text


def stopWordRemoval(text,custom):
	text = word_tokenize(text)
	text = ' '.join([word for word in text if word not in custom])
	return text


def lemData(text):
	lemmatizer = WordNetLemmatizer()
	
	text = word_tokenize(text)
	newText = []
	for word in text:
		newText.append(lemmatizer.lemmatize(word))
	return ' '.join(newText)

def get_tf_idx_matrix(doc):
	tfidf = TfidfVectorizer(stop_words="english", max_features= 1000,  smooth_idf=True)
	vector =tfidf.fit_transform(doc)
	X = vector.toarray()
	pd.DataFrame(X)	
	return X, tfidf

def get_truncatedSVD_outputs(X,tfidf):	
	feature_length = len(X[0])	
	if feature_length>5:
		n_components = 5
	elif feature_length<2:
		return 0, []
	else:
		n_components = feature_length-1	
		svd_model = TruncatedSVD(n_components=n_components, algorithm='randomized', n_iter=100, random_state=122)
	svd_model.fit(X)
	num_of_topics = len(svd_model.components_)	
	terms = tfidf.get_feature_names()
	topics = []	
	for i, comp in enumerate(svd_model.components_):
		terms_comp = zip(terms, comp)
		sorted_terms = sorted(terms_comp, key= lambda x:x[1], reverse=True)[:7]
		for t in sorted_terms:
			topics.append(t[0])	
	final_topic_list = [topics[i:i+8] for i in range(0, len(topics), 8)]	
	return num_of_topics,final_topic_list


def preprocess(text):
	tokenized_doc = clean_doc(text)
	stopwords_set = set(stopwords.words('english'))
	custom = list(stopwords_set)+list(punctuation)
	tokenized_doc = stopWordRemoval(tokenized_doc,custom)
	tokenized_doc = lemData(tokenized_doc)	
	return tokenized_doc

def flow_truncatedSVD_model(text):	
	tokenized_doc = clean_doc(text)	
	stopwords_set = set(stopwords.words('english'))
	custom = list(stopwords_set)+list(punctuation)
	tokenized_doc = stopWordRemoval(tokenized_doc,custom)
	
	tokenized_doc = lemData(tokenized_doc)	
	X,tfidf = get_tf_idx_matrix([tokenized_doc])	
	num_of_topics,final_topic_list = get_truncatedSVD_outputs(X,tfidf)	
	return num_of_topics,final_topic_list

# text = "\n\nI am sure some bashers of Pens fans are pretty confused about the lack\nof any kind of posts about the recent Pens massacre of the Devils. Actually,\nI am  bit puzzled too and a bit relieved. However, I am going to put an end\nto non-PIttsburghers' relief with a bit of praise for the Pens. Man, they\nare killing those Devils worse than I thought. Jagr just showed you why\nhe is much better than his regular season stats. He is also a lot\nfo fun to watch in the playoffs. Bowman should let JAgr have a lot of\nfun in the next couple of games since the Pens are going to beat the pulp out of Jersey anyway. I was very disappointed not to see the Islanders lose the final\nregular season game.          PENS RULE!!!\n "# num_of_topics,final_topic_list = flow_truncatedSVD_model(text)# print(num_of_topics)
# print(final_topic_list)# for x in final_topic_list:
# 	print(type(x))
# 	print(x)

