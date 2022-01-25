import nltk
words = set(nltk.corpus.words.words())
import re 

def clean_text_html(text):
  Tag_re=re.compile(r'<[^>]+>')
  text=Tag_re.sub('', text)
  return text

def clean_meaningless(text):
    return " ".join(w for w in nltk.wordpunct_tokenize(text) \
                    if w.lower() in words or not w.isalpha())


