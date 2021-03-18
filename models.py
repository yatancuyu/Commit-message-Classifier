import pandas as pd
import numpy as np

from tokenizers import StemTokenizer, LemmaTokenizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, f1_score, accuracy_score


def get_dataset():
    dataset = pd.read_csv('dataset.csv',
                          keep_default_na=False)

    X = dataset['Commit Message'].tolist()
    y = 1 * dataset['IsFix'].to_numpy()
    return X, y


def get_results(model):
    X, y = get_dataset()

    clf_models = {
        'logreg': LogisticRegression(),
        'decision_tree': DecisionTreeClassifier(max_depth=6),
        'bayes': MultinomialNB(),
        'svc': SVC()
    }
    clf_model = clf_models[model]

    MAX_DF = 0.5
    MIN_DF = 5
    NGRAM_RANGE = (1, 2)
    vect_models = [
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        use_idf=False),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        binary=True),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        tokenizer=LemmaTokenizer()),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        tokenizer=LemmaTokenizer(),
                        use_idf=False),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        tokenizer=LemmaTokenizer(),
                        binary=True),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        tokenizer=StemTokenizer()),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        tokenizer=StemTokenizer(),
                        use_idf=False),
        TfidfVectorizer(ngram_range=NGRAM_RANGE,
                        max_df=MAX_DF,
                        min_df=MIN_DF,
                        tokenizer=StemTokenizer(),
                        binary=True),
    ]

    statistics = np.zeros((3, 3), dtype=float)

    for i, vectorizer in enumerate(vect_models):
        train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.3, random_state=0)

        vectorizer.fit(X)
        train_X_Tfidf = vectorizer.transform(train_X)
        test_X_Tfidf = vectorizer.transform(test_X)

        clf_model.fit(train_X_Tfidf, train_y)

        prediction = clf_model.predict(test_X_Tfidf)
        statistics[i // 3, i % 3] = f1_score(test_y, prediction, average='macro')

    return statistics
