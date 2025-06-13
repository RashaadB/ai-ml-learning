## Welcome to Course5: Natural Language processing 
- In this course we will focus on Text Data Analysis, NLP Text vectorization, Distributed representations, machine translation, sequence models, attenction mechanism

## Text Data Analysis Folder

`NLP_intro.ipynb`
- what is NLP
- Levels of NLP
- components of NLP
    - NLU
    - NLG
    - DM
- Models used for NLP
- Challenges in NLP

`text_preprocessing_intro.ipynb`
- NLP Pipeline
- Text cleaning and text processing
    - tokenization
    - Text cleaning
    - stemming and lemmatiaztion
    - parts of speech tagging
    - named entity recognition
- feature engineering
- sentiment analysis

`text_preprocessing.ipynb`
- Data cleaning steps
    - Tokenization
    - Changing case
    - Spelling correction
    - POS Tagging
    - Named entity recognition (NER)
    - Stemming and Lemmatization
- Stemming
- Lemmatization
- Noise entity removal
- Remove stopwords
- Remove urls
- Remove punctuations
- Remove emoticons

`data_exploration_visualization.ipynb`
- Data extraction
- Check the files available in the corpus
- Use the 'Canon_G3.txt'
- Data exploration and visualization
- Basic exploration
- Detailed exploration
- Word frequency analysis
- Wordcloud generation

`text_classification.ipynb`
- Data acquisiton
- Feature extraction
- Model development
- Evaluation

`sentiment_analysis.ipynb`
- Sentiment analysis using TextBlob
- Sentiment analysis using VADER
- Sentiment analysis using flair package

## NLP Text Vectorization Folder
`nlp_text_vectorization.ipynb`
- Understanding Feature Vectors
- Word Analogies
- Distance Calculation
- Text Vectorization
- One Hot encoding
- Bag of words & Terms
- TF-IDF

`one_hot_encoding.ipynb`
<!-- The goal is to implement one-hot encoding for text data representation. The notebook covers acquiring and cleaning data, generating a vocabulary, and creating a one-hot encoded matrix. It concludes with displaying the matrix, showcasing how categorical text data is transformed into numerical format for machine learning tasks. -->

`bag_of_words.ipynb`
<!-- The goal is to implement the Bag-of-Words model for text data representation. The notebook introduces the concept and demonstrates its implementation using scikit-learn's CountVectorizer to transform text into numerical vectors for machine learning applications. -->

`tf_idf.ipynb`
<!-- The goal is to demonstrate the TF-IDF (Term Frequency-Inverse Document Frequency) method for text representation. The notebook introduces the concept and shows how to use scikit-learn's TfidfVectorizer to convert text into weighted numerical vectors, emphasizing the importance of terms in a corpus. -->

`word_two_vec.ipynb`
<!--  The goal is to provide a comprehensive understanding of Word2Vec, covering its key concepts, architecture, and working principles, including CBOW and Skip-gram models. It explains training, vector representation, loss functions, benefits, applications, and limitations, followed by practical implementation. The code example demonstrates library imports, corpus tokenization, model training, extracting word vectors, finding similar words, and visualizing word embeddings.-->
