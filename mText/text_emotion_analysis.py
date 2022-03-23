import json, os, sys, pickle
import spacy
nlp = spacy.load("en_core_web_sm")

from textblob import TextBlob
from allennlp.predictors.predictor import Predictor
import allennlp_models.tagging

from .truncatedSVD import flow_truncatedSVD_model

import nltk
# if not os.path.isdir(/home/gayatri/adEnv/lib/python3.8/site-packages/nltk):		
nltk.download('punkt')
nltk.download('brown')
nltk.download('stopwords')


def get_pos_ner(text):
	doc = nlp(text)
	pos_list = []

	for token in doc:
		text_obj = {}
		text_obj['token'] = token.text
		text_obj['lemma'] = token.lemma_
		text_obj['pos'] = token.pos_
		text_obj['tag'] = token.tag_
		text_obj['dep'] = token.dep_
		
		pos_list.append(text_obj)

	ner_list = []
	for ent in doc.ents:
		text_obj = {}
		text_obj['token'] = ent.text
		text_obj['start'] = ent.start_char
		text_obj['end'] = ent.end_char
		text_obj['ner'] = ent.label_
		text_obj['desc'] = str(spacy.explain(ent.label_))
		
		ner_list.append(text_obj)

	return pos_list, ner_list


def list_to_string(l):
	final_string = ""
	count = 0
	for i in l:
		if count == 0:
			final_string = str(i)
		else:
			final_string = final_string + ","+str(i)
		count +=1

	return final_string


class TextEmotion:

	def __init__(self):
		module_dir = os.path.dirname(__file__)
		filename = 'assets/finalized_model.sav'

		with open(os.path.join(module_dir, filename), 'rb') as file:
			self.model = pickle.load(file)

	def predict_text_emotion(self, obj_maudio):
		"""
		Text Emotion:

		Input: obj_maudio
			obj_maudio.transcript

		OutPut:
			polarity
			text_emotion
			text_ner
			text_pos
			topics
		"""
		text = obj_maudio.transcript
		if text:
			#text emotion
			emotion = self.model.predict([text])[0]

			#text POS/NER
			pos_list, ner_list = get_pos_ner(text)

			obj_maudio.text_emotion = emotion
			obj_maudio.text_ner = ner_list
			obj_maudio.text_pos = pos_list
			obj_maudio.save()

		return

	def text_sentiment_analysis(self, obj_maudio):
		"""
		Text sentiment:

		Input: obj_maudio
			obj_maudio.transcript

		OutPut:
			polarity
		"""
		text = obj_maudio.transcript
		text_analysis_data = {}

		if text:
			#text sentiment
			# predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/basic_stanford_sentiment_treebank-2020.06.09.tar.gz")

			blob = TextBlob(text)
			text_sentiment_textblob = blob.sentiment.polarity
			obj_maudio.polarity = text_sentiment_textblob
			obj_maudio.save()

			# text_sentiment_allennlp = predictor.predict(text)
			# text_analysis_data['text_sentiment_textblob'] = text_sentiment_textblob
			# text_analysis_data['text_sentiment_allennlp'] = text_sentiment_allennlp["label"]

			# sentences_sentiment = []

			# for sentence in blob.sentences:
			# 	sentence_str = str(sentence)
			# 	sentiment_textblob = sentence.sentiment.polarity
			# 	sentiment_textblob_allennlp = predictor.predict(sentence_str)
			# 	sentiment_textblob_allennlp = sentiment_textblob_allennlp['label']
			# 	obj = {"sentence":sentence_str,"sentiment_textblob":sentiment_textblob,"sentiment_textblob_allennlp":sentiment_textblob_allennlp}
			# 	sentences_sentiment.append(obj)
			# text_analysis_data['sentences_sentiment'] = sentences_sentiment
			# print("text_analysis_data: ", text_analysis_data)

		return


	def predict_topics(self, obj_maudio):

		"""
		Text Emotion:

		Input: obj_maudio
			obj_maudio.transcript

		OutPut:
			topics
		"""
		text = obj_maudio.transcript
		response = {}
		try:
			if text:
				num_of_topics, final_topic_list = flow_truncatedSVD_model(text)

				final_topic_list_strings = []

				for topics in final_topic_list:
					final_string = list_to_string(topics)
					final_topic_list_strings.append(final_string)

				response["truncatedSVD"] = {"model":"TruncatedSVD","num_of_topics":num_of_topics,"final_topic_list":final_topic_list_strings}

				obj_maudio.topics = final_topic_list_strings and final_topic_list_strings[0]
				obj_maudio.save()
		
		except Exception as e:
			print(e)

		




		
		
