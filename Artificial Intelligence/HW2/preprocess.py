import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.tokenize.toktok import ToktokTokenizer
import string

def remove_stopwords(text: str) -> str:
    '''
    E.g.,
        text: 'Here is a dog.'
        preprocessed_text: 'Here dog.'
    '''
    stop_word_list = stopwords.words('english')
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stop_word_list]
    preprocessed_text = ' '.join(filtered_tokens)

    return preprocessed_text

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()


import string
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import PorterStemmer

# Ensure the stemmer is defined
stemmer = PorterStemmer()

def preprocessing_function(text: str) -> str:
    # Remove stopwords first
    preprocessed_text = remove_stopwords(text)
    
    # Remove punctuation
    preprocessed_text = preprocessed_text.translate(str.maketrans('', '', string.punctuation))
    
    # Convert text to lowercase
    preprocessed_text = preprocessed_text.lower()

    # Apply stemming
    tokenizer = ToktokTokenizer()
    tokens = tokenizer.tokenize(preprocessed_text)
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    preprocessed_text = ' '.join(stemmed_tokens)

    return preprocessed_text
