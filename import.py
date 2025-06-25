# --- Imports ---
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re
import nltk
import spacy

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel, LsiModel, TfidfModel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- Setup ---
plt.rcParams['figure.figsize'] = (12, 8)
default_plot_colour = "#00bfbf"

# --- Load Data ---
data = pd.read_csv("fake_news_data.csv")

# Plot class distribution
data['fake_or_factual'].value_counts().plot(kind='bar', color=default_plot_colour)
plt.title('Count of Article Classification')
plt.ylabel('# of Articles')
plt.xlabel('Classification')
plt.show()

# --- Text Cleaning ---
# remove location tags (e.g. "LONDON - ")
data['text_clean'] = data['text'].apply(lambda x: re.sub(r"^[^-]*-\s*", "", x))
data['text_clean'] = data['text_clean'].str.lower()
data['text_clean'] = data['text_clean'].apply(lambda x: re.sub(r"[^\w\s]", "", x))

# remove stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))
data['text_clean'] = data['text_clean'].apply(lambda x: ' '.join([word for word in x.split() if word not in stop_words]))

# tokenization and lemmatization
nltk.download('punkt')
nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()
data['text_clean'] = data['text_clean'].apply(lambda x: word_tokenize(x))
data['text_clean'] = data['text_clean'].apply(lambda tokens: [lemmatizer.lemmatize(token) for token in tokens])

# --- Unigrams ---
tokens_clean = sum(data['text_clean'], [])
unigrams = (pd.Series(nltk.ngrams(tokens_clean, 1)).value_counts()).reset_index()[:10]
unigrams.columns = ['index', 'count']
unigrams['token'] = unigrams['index'].apply(lambda x: x[0])

sns.barplot(x='count', y='token', data=unigrams, palette=[default_plot_colour])
plt.title('Most Common Unigrams After Preprocessing')
plt.show()

# --- Bigrams ---
bigrams = pd.Series(nltk.ngrams(tokens_clean, 2)).value_counts()[:10]
print("Top Bigrams:\n", bigrams)

# --- Sentiment Analysis ---
vader = SentimentIntensityAnalyzer()
data['vader_sentiment_score'] = data['text'].apply(lambda x: vader.polarity_scores(x)['compound'])

def get_sentiment(score):
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

data['vader_sentiment_label'] = data['vader_sentiment_score'].apply(get_sentiment)

# Plot sentiment counts
data['vader_sentiment_label'].value_counts().plot(kind='bar', color=default_plot_colour)
plt.title("Sentiment Distribution")
plt.show()

sns.countplot(x='fake_or_factual', hue='vader_sentiment_label', data=data, palette="hls")
plt.title("Sentiment by News Type")
plt.show()

# --- LDA Topic Modeling on Fake News ---
fake_news_text = data[data['fake_or_factual'] == "Fake News"]['text_clean'].reset_index(drop=True)
dictionary_fake = corpora.Dictionary(fake_news_text)
doc_term_fake = [dictionary_fake.doc2bow(text) for text in fake_news_text]

# LDA coherence scores
coherence_vals = []
min_topics = 2
max_topics = 11
for num in range(min_topics, max_topics + 1):
    model = gensim.models.LdaModel(doc_term_fake, num_topics=num, id2word=dictionary_fake)
    coherence = CoherenceModel(model=model, texts=fake_news_text, dictionary=dictionary_fake, coherence='c_v')
    coherence_vals.append(coherence.get_coherence())

plt.plot(range(min_topics, max_topics + 1), coherence_vals)
plt.xlabel("Number of Topics")
plt.ylabel("Coherence Score")
plt.title("LDA Coherence Scores")
plt.show()

# Train final LDA model
lda_model = gensim.models.LdaModel(corpus=doc_term_fake, id2word=dictionary_fake, num_topics=6)
print("LDA Topics:\n", lda_model.print_topics())

# --- TF-IDF and LSA ---
def tfidf_corpus(doc_term_matrix):
    tfidf = TfidfModel(corpus=doc_term_matrix, normalize=True)
    return tfidf[doc_term_matrix]

def get_coherence_scores(corpus, dictionary, texts, min_topics, max_topics):
    scores = []
    for num in range(min_topics, max_topics + 1):
        model = LsiModel(corpus=corpus, num_topics=num, id2word=dictionary)
        coherence = CoherenceModel(model=model, texts=texts, dictionary=dictionary, coherence='c_v')
        scores.append(coherence.get_coherence())
    plt.plot(range(min_topics, max_topics + 1), scores)
    plt.xlabel("Number of Topics")
    plt.ylabel("Coherence Score")
    plt.title("LSA Coherence Scores")
    plt.show()

tfidf_corpus_fake = tfidf_corpus(doc_term_fake)
get_coherence_scores(tfidf_corpus_fake, dictionary_fake, fake_news_text, min_topics=2, max_topics=11)

# Final LSA model
lsa_model = LsiModel(tfidf_corpus_fake, id2word=dictionary_fake, num_topics=5)
print("LSA Topics:\n", lsa_model.print_topics())

# --- Classification: Logistic Regression & SVM ---
X = [' '.join(tokens) for tokens in data['text_clean']]
Y = data['fake_or_factual']

# Vectorization
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_vectorized, Y, test_size=0.3, random_state=42)

# Logistic Regression
lr = LogisticRegression(random_state=0).fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLogistic Regression Accuracy:", accuracy_score(y_pred_lr, y_test))
print(classification_report(y_test, y_pred_lr))

# SVM Classifier
svm = SGDClassifier().fit(X_train, y_train)
y_pred_svm = svm.predict(X_test)
print("\nSVM Accuracy:", accuracy_score(y_pred_svm, y_test))
print(classification_report(y_test, y_pred_svm))

