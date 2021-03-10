import re
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

stp = stopwords.words('indonesian')
factory = StemmerFactory()
stemmer = factory.create_stemmer()

# Preprocessing
def lower(text):
    # Case Folding
    return text.lower()


def remove_punctuation(text):
    # Happy Emoticons
    emoticons_happy = set([
    ':-)', ':)', ';)', ':o)', ':]', ':3', ':c)', ':>', '=]', '8)', '=)', ':}',
    ':^)', ':-D', ':D', ':d', '8-D', '8D', 'x-D', 'xD', 'X-D', 'XD', '=-D', '=D',
    '=-3', '=3', ':-))', ":'-)", ":')", ':*', ':^*', '>:P', ':-P', ':P', 'X-P',
    'x-p', 'xp', 'XP', ':-p', ':p', '=p', ':-b', ':b', '>:)', '>;)', '>:-)',
    '<3'
    ])
 
    # Sad Emoticons
    emoticons_sad = set([
    ':L', ':-/', '>:/', ':S', '>:[', ':@', ':-(', ':[', ':-||', '=L', ':<',
    ':-[', ':-<', '=\\', '=/', '>:(', ':(', '>.<', ":'-(", ":'(", ':\\', ':-c',
    ':c', ':{', '>:\\', ';('
    ])
 
    # All emoticons (happy + sad)
    emoticons = emoticons_happy.union(emoticons_sad)

    text = ' '.join([word for word in text.split() if word not in emoticons])

    text = re.sub(r'@[\w]*', ' ', text)

    text = re.sub(r'\w+:\/{2}[\d\w-]+(\.[\d\w-]+)*(?:(?:\/[^\s/]*))*', ' ', text)

    text = re.sub(r'^https?:\/\/.*[\r\n]*', ' ', text)

    text = re.sub(r'^RT[\s]+', ' ', text)  

    text = text.lower()  

    text = re.sub(r'[^\w\s]+', ' ', text)

    text = re.sub(r'[0-9]+', ' ', text)

    text = re.sub(r'_', ' ', text)

    text = re.sub(r'\$\w*', ' ', text)
    
    return text


def remove_stopwords(text):
    text = ' '.join([word for word in text.split() if word not in stp])
    return text


def stem_text(text):
    text = ' '.join([stemmer.stem(word) for word in text.split()])
    return text


# Kalimat Testing
def preprocess_data(text):
    text = remove_punctuation(text)
    text = remove_stopwords(text)
    text = stem_text(text)
    return text

