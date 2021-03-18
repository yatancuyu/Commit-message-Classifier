import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer

TOKEN_RE = re.compile(r'[\w]+')


def tokenize_text_simple_regex(txt, min_token_size=3):
    txt = txt.lower()
    all_tokens = TOKEN_RE.findall(txt)
    stop_words = set(stopwords.words("english"))

    without_stop_words = [token for token in all_tokens if token not in stop_words]
    return [token for token in without_stop_words if len(token) >= min_token_size]


class LemmaTokenizer:
    def __init__(self):
        self.wnl = WordNetLemmatizer()

    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in tokenize_text_simple_regex(doc)]


class StemTokenizer:
    def __init__(self):
        self.ps = PorterStemmer()

    def __call__(self, doc):
        return [self.ps.stem(word) for word in tokenize_text_simple_regex(doc)]
