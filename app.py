# Flask Packages
from flask import Flask,render_template,request,url_for
from flask_bootstrap import Bootstrap 
from flask_uploads import UploadSet,configure_uploads,IMAGES,DATA,ALL
from flask_sqlalchemy import SQLAlchemy 

from werkzeug import secure_filename
import os
import datetime
import time

# EDA Packages
import pandas as pd 
import numpy as np 

## LDA packages
import gensim
from gensim.utils import simple_preprocess
from gensim.parsing.preprocessing import STOPWORDS
from nltk.stem import WordNetLemmatizer, SnowballStemmer
from nltk.stem.porter import *
from gensim.models import CoherenceModel
np.random.seed(2018)
import nltk
nltk.download('wordnet')
import matplotlib.pyplot as plt
from gensim import corpora, models


app = Flask(__name__)
Bootstrap(app)
db = SQLAlchemy(app)

# Configuration for File Uploads
files = UploadSet('files',ALL)
app.config['UPLOADED_FILES_DEST'] = 'static/uploadsDB'
configure_uploads(app,files)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///static/uploadsDB/filestorage.db'

# Saving Data To Database Storage
class FileContents(db.Model):
	id = db.Column(db.Integer,primary_key=True)
	name = db.Column(db.String(300))
	modeldata = db.Column(db.String(300))
	data = db.Column(db.LargeBinary)


@app.route('/')
def index():
	return render_template('index.html')

# Route for our Processing and Details Page
@app.route('/dataupload',methods=['GET','POST'])
def dataupload():
	if request.method == 'POST' and 'csv_data' in request.files:
		
		num_topics = request.form['num_topics']
		
		file = request.files['csv_data']
		filename = secure_filename(file.filename)
        # os.path.join is used so that paths work in every operating system
        # file.save(os.path.join("wherever","you","want",filename))
		file.save(os.path.join('static/uploadsDB',filename))
		fullfile = os.path.join('static/uploadsDB',filename)

        # EDA function
		raw = pd.read_csv(os.path.join('static/uploadsDB',filename))
		col = list(raw.columns)[-1]
		
        ## Preprocessing
        # data cleaning
		
		def clean_data(rawdata, col):
        # 1. Drop NA text rows
			rawdata = rawdata.dropna(subset=[col])

        # 2. Drop duplicated texts
			rawdata["duplicated"] = rawdata.duplicated(subset = col, keep = "first") #duplicates are marked as True
			rawdata = rawdata[rawdata['duplicated'] == False].copy()

    # 3. Remove url links
			rawdata[col] = rawdata[col].str.replace(r'http\S+|www.\S+', '', case=False)

    # 4. Drop empty strings
			rawdata[col].replace('', np.nan, inplace=True)
			rawdata = rawdata.dropna(subset = [col])

			return rawdata
        
		df = clean_data(raw, col)
		
        # preprocess (tokenize, remove stopwords, remove words <= 3 characters)
		def preprocess(text):
			result = []
			for token in gensim.utils.simple_preprocess(text):
				if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 3:
					result.append(token)
			return result
		def lemmatize(text):
			return WordNetLemmatizer().lemmatize(text, pos='v')
		
		def preprocess_lemmatize(text):
			result = []
			for token in gensim.utils.simple_preprocess(text):
				if token not in gensim.parsing.preprocessing.STOPWORDS and len(token) >= 3:
					result.append(lemmatize(token))
			return result
		
		processed_doc = df[col].apply(lambda x: preprocess_lemmatize(x))
		

		def bagofwords(tokens, no_below = 15, no_above = 0.5, keep_n = 10000 ):
			dictionary = gensim.corpora.Dictionary(tokens)

            #filter the vocab
			dictionary.filter_extremes(no_below, no_above, keep_n)

            # bag of words. vocab and its frequency
			bow_corpus = [dictionary.doc2bow(doc) for doc in tokens]
			tfidf = models.TfidfModel(bow_corpus)
			corpus_tfidf = tfidf[bow_corpus]
			
			return dictionary, bow_corpus, tfidf, corpus_tfidf

        # select which processed_doc you would like to input to bag of words
		dictionary, bow_corpus, tfidf, corpus_tfidf = bagofwords(processed_doc, no_below = 15, no_above = 0.5, keep_n = 10000)


        ## LDA with normal bag of words
		num_topics = int(num_topics)
		lda_model = gensim.models.LdaMulticore(bow_corpus, num_topics=num_topics, id2word=dictionary, passes=2, workers=2)
		output1 = ""
		for idx, topic in lda_model.print_topics(-1):
			output1 += "\n"
			output1 += 'Topic: {} Words: {}'.format(idx, topic)
			output1 += "\n"


        ## LDA using tfidf
		num_topics = int(num_topics)
		lda_model_tfidf = gensim.models.LdaMulticore(corpus_tfidf, num_topics=num_topics, id2word=dictionary, passes=2, workers=4)
		outputL = []
		for idx, topic in lda_model_tfidf.print_topics(-1):
			outputL.append('Topic: {} Word: {}'.format(idx, topic))
		
		
		def make_bigrams(texts):
			return [bigram_mod[doc] for doc in texts]
		
		data_words = list(processed_doc)
		bigram = gensim.models.Phrases(data_words, min_count=5, threshold=100)
		bigram_mod = gensim.models.phrases.Phraser(bigram)	
		data_words_bigrams = make_bigrams(data_words)
		bi_dictionary, bi_bow_corpus, bi_tfidf, bi_corpus_tfidf = bagofwords(data_words_bigrams, no_below = 15, no_above = 0.5, keep_n = 10000)
		
		num_topics=num_topics
		lda_model_tfidf_bi = gensim.models.LdaMulticore(bi_corpus_tfidf, num_topics=num_topics, id2word=bi_dictionary, passes=2, workers=4)      
		outputB = []
		for idx, topic in lda_model_tfidf_bi.print_topics(-1):
			outputB.append('Topic: {} Word: {}'.format(idx, topic))
		
		## add topic assignments to original data frame
		def format_topics_sentences(ldamodel,corpus, texts):
			sent_topics_df = pd.DataFrame()

    # Get main topic in each document
			for i, row in enumerate(ldamodel[corpus]):
				row = sorted(row, key=lambda x: (x[1]), reverse=True)
		        # Get the Dominant topic, Perc Contribution and Keywords for each document
				for j, (topic_num, prop_topic) in enumerate(row):
					if j == 0:  # => dominant topic
						wp = ldamodel.show_topic(topic_num)
						topic_keywords = ", ".join([word for word, prop in wp])
						sent_topics_df = sent_topics_df.append(pd.Series([int(topic_num), round(prop_topic,4), topic_keywords]), ignore_index=True)
					else:
						break
			sent_topics_df.columns = ['Main Topic', 'Probability', 'Topic Keywords']
			
			contents = pd.Series(texts)
			sent_topics_df = pd.concat([sent_topics_df, contents], axis=1)
			
			return(sent_topics_df)
		texts = df[col].to_list()
		df_topic_sents_keywords = format_topics_sentences(ldamodel=lda_model_tfidf_bi, corpus=bi_corpus_tfidf, texts=texts)
		
		raw["duplicated"] = raw.duplicated(keep = "first")
		raw = raw[raw['duplicated'] == False].copy()
		raw["text"] = raw[col].str.replace(r'http\S+|www.\S+', '', case=False)
		raw["text"].replace('', np.nan, inplace=True)
		raw = raw.dropna(subset=[col])
		raw = pd.merge(raw,df_topic_sents_keywords,left_on='text', right_on=0, how='left')
		raw = raw.drop(columns = ["duplicated", "text",0])
		
		raw.to_csv('static/uploadsDB/result.csv')
		
		df_shape = raw.shape
		df_targetname = col
		
	return render_template('details.html',
						df_shape=df_shape,
						df_targetname =df_targetname,
						fullfile = fullfile,
						result1 = outputL,
						result2 = outputB,
						dfplot = raw.tail(n=10),
						num_topics = str(num_topics))

if __name__ == '__main__':
    app.run(debug=True)





# Jesus Saves @ JCharisTech