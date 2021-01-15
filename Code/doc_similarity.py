import nltk
import pandas as pd
nltk.download('stopwords')
nltk.download('punkt')
from nltk.tokenize import WordPunctTokenizer
from nltk.tokenize import sent_tokenize, word_tokenize

from nltk.stem import PorterStemmer 
from nltk.corpus import stopwords
import string


#loading the files
text1=open(r"E:\MS_Studies\sem2\566\hw1\uiuc.txt","rt", encoding="utf8")
uiuc=text1.read()
text1.close()

text2=open(r"E:\MS_Studies\sem2\566\hw1\uis.txt","rt", encoding="utf8")
uis=text2.read()
text2.close()

text3=open(r"E:\MS_Studies\sem2\566\hw1\mit.txt","rt", encoding="utf8")
mit=text3.read()
text3.close()

text4=open(r"E:\MS_Studies\sem2\566\hw1\stanford.txt","rt", encoding="utf8")
stanford=text4.read()
text4.close()

text5=open(r"E:\MS_Studies\sem2\566\hw1\tesla.txt","rt", encoding="utf8")
tesla=text5.read()
text5.close()

text6=open(r"E:\MS_Studies\sem2\566\hw1\uic.txt","rt", encoding="utf8")
uic=text6.read()
text6.close()




# step a: function to remove punctuation/apostrophe/convert to lower
def removal(textfile):
    y=textfile
    ##step 1: converting to lower case
    y=y.lower()
    
    #step 2: replacing apostraphe
    y=y.replace("'", " ")
    
    # step 3: remove_punctuation
    y=y.translate(str.maketrans(string.punctuation,' '*len(string.punctuation)))
    return y
    

#step b: call to function removal
uic=removal(uic)
uiuc=removal(uiuc)
uis=removal(uis)
mit=removal(mit)
stanford=removal(stanford)
tesla=removal(tesla)


def process(text):
    x=text
    #step4: tokenization ( split using word tokenizer)
    #tk = word_tokenize()  
    x=word_tokenize(x)

    # step5: filtering out tokens that are not alphabetic
    x= [c for c in x if c.isalpha()]
    
    #step6: removing stop words
    list_stop_words = set(stopwords.words('english'))
    x      = [w for w in x if not w in list_stop_words]
    
    # step7: Stemming
    ps = PorterStemmer()
    x= [ps.stem(c) for c in x]
    
    return x
 
 
 uic=process(uic)
uiuc=process(uiuc)
uis=process(uis)
mit=process(mit)
stanford=process(stanford)
tesla=process(tesla)



#function to find jaccard similarity
def get_jaccard_sim(str1, str2): 
    a = set(str1) 
    b = set(str2)
    c = a.intersection(b)
    print(c)
    return float(len(c)) / (len(a) + len(b) - len(c))
    
    
uiuc_jac = get_jaccard_sim(uic,uiuc)
uis_jac = get_jaccard_sim(uic,uis)
mit_jac = get_jaccard_sim(uic,mit)
stanford_jac = get_jaccard_sim(uic,stanford)
tesla_jac = get_jaccard_sim(uic,tesla)

list_jac=[uiuc_jac,uis_jac,mit_jac,stanford_jac,tesla_jac]


# Detokenizing the documents
from nltk.tokenize.treebank import TreebankWordDetokenizer
uic1=TreebankWordDetokenizer().detokenize(uic)
uiuc1=TreebankWordDetokenizer().detokenize(uiuc)
uis1=TreebankWordDetokenizer().detokenize(uis)
mit1=TreebankWordDetokenizer().detokenize(mit)
stanford1=TreebankWordDetokenizer().detokenize(stanford)
tesla1=TreebankWordDetokenizer().detokenize(tesla)


#vectorizing and finding cosine similarity
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
documents=[uic1,uiuc1,uis1,mit1,stanford1,tesla1]
tfidf = TfidfVectorizer().fit_transform(documents)
pairwise_similarity = tfidf * tfidf.T

##finding similarity of UIC with others using cosine_similarity function available in the SKlearn 
similarity=cosine_similarity(tfidf[0:1],tfidf)



