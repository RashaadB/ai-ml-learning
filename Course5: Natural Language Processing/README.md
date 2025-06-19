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

## Distributed Representation Folder
`distributed_representation.ipynb`
- Draw backs of Fundamental Vectorization
- Distributed Representations
- Word Embeddings
- Word2Vec
- GloVe Word Representation
- FastText
- Doc2Vec

`pre_trained_word_embeddings.ipynb`
<!-- The objective here is to set up an environment, implement or load a model architecture, and then use a pre-trained embedding model for generating embeddings that can be applied to further tasks in NLP or machine learning applications. -->

`doc_vectors_using_average_spacy.ipynb`
<!-- The goal is to generate document vectors using SpaCy by processing text data and calculating average vector representations. The notebook demonstrates how to extract linguistic annotations and token-level vectors for further analysis. This approach enables efficient handling of text data for downstream NLP tasks. -->

`training_doc2vec_gensim.ipynb`
<!-- The goal is to implement paragraph vector models for text representation by preparing the data and leveraging two key approaches: Distributed Bag of Words (DBoW) and Distributed Memory (PV-DM). These methods capture semantic information to create meaningful paragraph embeddings for downstream NLP tasks. -->

## Machine Translation Folder

`language_preprocessing.ipynb`
<!-- The goal is to guide language preprocessing by covering essential steps like tokenization, sequencing, padding, and vocabulary indexing. The process includes text, sentence, and word tokenization, followed by converting text into sequences, adding padding for uniform input lengths, and building a vocabulary index. Example code and explanations make it practical for NLP applications. -->

`machine_translation.ipynb`
<!-- The goal is to build a machine translation pipeline by leveraging embeddings to translate an English dictionary to French. It involves loading necessary libraries and embeddings, working with embedding vectors, and using cosine similarity to measure semantic similarity. Additionally, gradient computation optimizes the transformation matrix for effective translation. -->

`machine_translation_Encoder_decoder.ipynb`
<!-- The goal is to implement a machine translation system using an encoder-decoder model. It involves preparing a suitable dataset, defining the model architecture, training it to translate between languages, and performing inference to generate translations. The conclusion summarizes the results and insights from the implementation. -->