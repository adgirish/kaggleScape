
# coding: utf-8

# Introduction
# ====
# **Natural Language Processing ** (NLP) is the task of making computers understand and produce human languages. 
# 
# And it always starts with the **corpus** i.e. *a body of text*. 
# 

# 
# What is a Corpus?
# ====
# 
# There are many corpora (*plural of corpus*) available in NLTK, lets start with an English one call the **Brown corpus**.
# 
# When using a new corpus in NLTK for the first time, downloads the corpus with the `nltk.download()` function, e.g. 
# 
# ```python
# import nltk
# nltk.download('brown')
# ```

# After its downloaded, you can import it as such:

# In[ ]:


from nltk.corpus import brown


# In[ ]:


brown.words() # Returns a list of strings


# In[ ]:


len(brown.words()) # No. of words in the corpus


# In[ ]:


brown.sents() # Returns a list of list of strings 


# In[ ]:


brown.sents(fileids='ca01') # You can access a specific file with `fileids` argument.


# 
# **Fast Facts:**
# 
# > The Brown Corpus of Standard American English was the first of the modern, computer readable, general corpora. It was compiled by W.N. Francis and H. Kucera, Brown University, Providence, RI. The corpus consists of one million words of American English texts printed in 1961.
# 
# (Source: [University of Essex Corpus Linguistics site](  https://www1.essex.ac.uk/linguistics/external/clmt/w3c/corpus_ling/content/corpora/list/private/brown/brown.html))
# 
# >  This corpus contains text from 500 sources, and the sources have been categorized by genre, such as news, editorial, and so on ... (for a complete list, see http://icame.uib.no/brown/bcm-los.html).
# 
# ![](http://)(Source: [NLTK book, Chapter 2.1.3](http://www.nltk.org/book/ch02.html))

# The actual `brown` corpus data is **packaged as raw text files**.  And you can find their IDs with: 

# In[ ]:


len(brown.fileids()) # 500 sources, each file is a source.


# In[ ]:


print(brown.fileids()[:100]) # First 100 sources.


# You can access the raw files with:

# In[ ]:


print(brown.raw('cb01').strip()[:1000]) # First 1000 characters.


# <br>
# You will see that **each word comes with a slash and a label** and unlike normal text, we see that **punctuations are separated from the word that comes before it**, e.g. 
# 
# > The/at General/jj-tl Assembly/nn-tl ,/, which/wdt adjourns/vbz today/nr ,/, has/hvz performed/vbn in/in an/at atmosphere/nn of/in crisis/nn and/cc struggle/nn from/in the/at day/nn it/pps convened/vbd ./.
# 
# <br>
# And we also see that the **each sentence is separated by a newline**:
# 
# > There/ex followed/vbd the/at historic/jj appropriations/nns and/cc budget/nn fight/nn ,/, in/in which/wdt the/at General/jj-tl Assembly/nn-tl decided/vbd to/to tackle/vb executive/nn powers/nns ./.
# > 
# > The/at final/jj decision/nn went/vbd to/in the/at executive/nn but/cc a/at way/nn has/hvz been/ben opened/vbn for/in strengthening/vbg budgeting/vbg procedures/nns and/cc to/to provide/vb legislators/nns information/nn they/ppss need/vb ./.
# 
# <br>
# That brings us to the next point on **sentence tokenization** and **word tokenization**.

# Tokenization
# ====
# 
# **Sentence tokenization** is the process of  *splitting up strings into “sentences”*
# 
# **Word tokenization** is the process of  *splitting up “sentences” into “words”*
# 
# Lets play around with some interesting texts,  the `singles.txt` from `webtext` corpus. <br>
# They were some  **singles ads** from  http://search.classifieds.news.com.au/
# 
# First, downoad the data with `nltk.download()`:
# 
# ```python
# nltk.download('webtext')
# ```
# 
# Then you can import with:

# In[ ]:


from nltk.corpus import webtext


# In[ ]:


webtext.fileids()


# In[ ]:


# Each line is one advertisement.
for i, line in enumerate(webtext.raw('singles.txt').split('\n')):
    if i > 10: # Lets take a look at the first 10 ads.
        break
    print(str(i) + ':\t' + line)


# # Lets zoom in on candidate no. 8

# In[ ]:


single_no8 = webtext.raw('singles.txt').split('\n')[8]
print(single_no8)


# # Sentence Tokenization
# <br>
# In NLTK, `sent_tokenize()` the default tokenizer function that you can use to split strings into "*sentences*". 
# <br>
# 
# It is using the [**Punkt algortihm** from Kiss and Strunk (2006)](http://www.mitpressjournals.org/doi/abs/10.1162/coli.2006.32.4.485).

# In[ ]:


from nltk import sent_tokenize, word_tokenize


# In[ ]:


sent_tokenize(single_no8)


# In[ ]:


for sent in sent_tokenize(single_no8):
    print(word_tokenize(sent))


# # Lowercasing
# 
# The CAPS in the texts are RATHER irritating although we KNOW the guy is trying to EMPHASIZE on something ;P
# 
# We can simply **lowercase them after we do `sent_tokenize()` and `word_tokenize()`**. <br>
# The tokenizers uses the capitalization as cues to know when to split so removing them before the calling the functions would be sub-optimal.

# In[ ]:


sent_tokenize(single_no8)


# In[ ]:


for sent in sent_tokenize(single_no8):
    # It's a little in efficient to loop through each word,
    # after but sometimes it helps to get better tokens.
    print([word.lower() for word in word_tokenize(sent)])
    # Alternatively:
    #print(list(map(str.lower, word_tokenize(sent))))


# In[ ]:


print(word_tokenize(single_no8))  # Treats the whole line as one document.


# # Tangential Note
# 
# Punkt is a statistical model so it applies the knowledge it has learnt from previous data. <br>
# Generally, it **works for most cases on well-formed texts** but if your data is  different e.g. user-generated noisy texts, you might have to retrain a new model. 
# ![](http://)
# E.g. if we look at candidate no. 9, we see that it's splitting on `y.o.` (its thinking that its the end of the sentnence) and not splitting on `&c.` (its thinking that its an abbreviation, e.g. `Mr.`, `Inc.`).

# In[ ]:


single_no9 = webtext.raw('singles.txt').split('\n')[9]
sent_tokenize(single_no9)


# Stopwords
# ====
# 
# **Stopwords** are non-content words that primarily has only grammatical function
# 
# In NLTK, you can access them as follows:

# In[ ]:


from nltk.corpus import stopwords

stopwords_en = stopwords.words('english')
print(stopwords_en)


# # Often we want to remove stopwords when we want to keep the "gist" of the document/sentence.
# 
# For instance, lets go back to the our `single_no8`

# In[ ]:


# Treat the multiple sentences as one document (no need to sent_tokenize)
# Tokenize and lowercase
single_no8_tokenized_lowered = list(map(str.lower, word_tokenize(single_no8)))
print(single_no8_tokenized_lowered)


# # Let's try to remove the stopwords using the English stopwords list in NLTK

# In[ ]:


stopwords_en = set(stopwords.words('english')) # Set checking is faster in Python than list.

# List comprehension.
print([word for word in single_no8_tokenized_lowered if word not in stopwords_en])


# # Often, we want to remove the punctuations from the documents too.
# 
# Since Python comes with "batteries included", we have string.punctuation

# In[ ]:


from string import punctuation
# It's a string so we have to them into a set type
print('From string.punctuation:', type(punctuation), punctuation)


# # Combining the punctuation with the stopwords from NLTK.

# In[ ]:


stopwords_en_withpunct = stopwords_en.union(set(punctuation))
print(stopwords_en_withpunct)


# # Removing stopwords with punctuations from Single no. 8

# In[ ]:


print([word for word in single_no8_tokenized_lowered if word not in stopwords_en_withpunct])


# # Using a stronger/longer list of stopwords
# 
# From the previous output, we have still dangly model verbs (i.e. 'could', 'wont', etc.).
# 
# We can combine the stopwords we have in NLTK with other stopwords list we find online.
# 
# Personally, I like to use `stopword-json` because it has stopwrds in 50 languages =) <br>
# https://github.com/6/stopwords-json

# In[ ]:


# Stopwords from stopwords-json
stopwords_json = {"en":["a","a's","able","about","above","according","accordingly","across","actually","after","afterwards","again","against","ain't","all","allow","allows","almost","alone","along","already","also","although","always","am","among","amongst","an","and","another","any","anybody","anyhow","anyone","anything","anyway","anyways","anywhere","apart","appear","appreciate","appropriate","are","aren't","around","as","aside","ask","asking","associated","at","available","away","awfully","b","be","became","because","become","becomes","becoming","been","before","beforehand","behind","being","believe","below","beside","besides","best","better","between","beyond","both","brief","but","by","c","c'mon","c's","came","can","can't","cannot","cant","cause","causes","certain","certainly","changes","clearly","co","com","come","comes","concerning","consequently","consider","considering","contain","containing","contains","corresponding","could","couldn't","course","currently","d","definitely","described","despite","did","didn't","different","do","does","doesn't","doing","don't","done","down","downwards","during","e","each","edu","eg","eight","either","else","elsewhere","enough","entirely","especially","et","etc","even","ever","every","everybody","everyone","everything","everywhere","ex","exactly","example","except","f","far","few","fifth","first","five","followed","following","follows","for","former","formerly","forth","four","from","further","furthermore","g","get","gets","getting","given","gives","go","goes","going","gone","got","gotten","greetings","h","had","hadn't","happens","hardly","has","hasn't","have","haven't","having","he","he's","hello","help","hence","her","here","here's","hereafter","hereby","herein","hereupon","hers","herself","hi","him","himself","his","hither","hopefully","how","howbeit","however","i","i'd","i'll","i'm","i've","ie","if","ignored","immediate","in","inasmuch","inc","indeed","indicate","indicated","indicates","inner","insofar","instead","into","inward","is","isn't","it","it'd","it'll","it's","its","itself","j","just","k","keep","keeps","kept","know","known","knows","l","last","lately","later","latter","latterly","least","less","lest","let","let's","like","liked","likely","little","look","looking","looks","ltd","m","mainly","many","may","maybe","me","mean","meanwhile","merely","might","more","moreover","most","mostly","much","must","my","myself","n","name","namely","nd","near","nearly","necessary","need","needs","neither","never","nevertheless","new","next","nine","no","nobody","non","none","noone","nor","normally","not","nothing","novel","now","nowhere","o","obviously","of","off","often","oh","ok","okay","old","on","once","one","ones","only","onto","or","other","others","otherwise","ought","our","ours","ourselves","out","outside","over","overall","own","p","particular","particularly","per","perhaps","placed","please","plus","possible","presumably","probably","provides","q","que","quite","qv","r","rather","rd","re","really","reasonably","regarding","regardless","regards","relatively","respectively","right","s","said","same","saw","say","saying","says","second","secondly","see","seeing","seem","seemed","seeming","seems","seen","self","selves","sensible","sent","serious","seriously","seven","several","shall","she","should","shouldn't","since","six","so","some","somebody","somehow","someone","something","sometime","sometimes","somewhat","somewhere","soon","sorry","specified","specify","specifying","still","sub","such","sup","sure","t","t's","take","taken","tell","tends","th","than","thank","thanks","thanx","that","that's","thats","the","their","theirs","them","themselves","then","thence","there","there's","thereafter","thereby","therefore","therein","theres","thereupon","these","they","they'd","they'll","they're","they've","think","third","this","thorough","thoroughly","those","though","three","through","throughout","thru","thus","to","together","too","took","toward","towards","tried","tries","truly","try","trying","twice","two","u","un","under","unfortunately","unless","unlikely","until","unto","up","upon","us","use","used","useful","uses","using","usually","uucp","v","value","various","very","via","viz","vs","w","want","wants","was","wasn't","way","we","we'd","we'll","we're","we've","welcome","well","went","were","weren't","what","what's","whatever","when","whence","whenever","where","where's","whereafter","whereas","whereby","wherein","whereupon","wherever","whether","which","while","whither","who","who's","whoever","whole","whom","whose","why","will","willing","wish","with","within","without","won't","wonder","would","wouldn't","x","y","yes","yet","you","you'd","you'll","you're","you've","your","yours","yourself","yourselves","z","zero"]}
stopwords_json_en = set(stopwords_json['en'])
stopwords_nltk_en = set(stopwords.words('english'))
stopwords_punct = set(punctuation)
# Combine the stopwords. Its a lot longer so I'm not printing it out...
stoplist_combined = set.union(stopwords_json_en, stopwords_nltk_en, stopwords_punct)

# Remove the stopwords from `single_no8`.
print('With combined stopwords:')
print([word for word in single_no8_tokenized_lowered if word not in stoplist_combined])


# # Stemming and Lemmatization
# 
# Often we want to map the different forms of the same word to the same root word, e.g. "walks", "walking", "walked" should all be the same as "walk".
# 
# The stemming and lemmatization process are hand-written regex rules written find the root word.
# 
#  - **Stemming**: Trying to shorten a word with simple regex rules
# 
#  - **Lemmatization**: Trying to find the root word with linguistics rules (with the use of regexes)
# 
# (See also: [Stemmers vs Lemmatizers](https://stackoverflow.com/q/17317418/610569) question on StackOverflow)
# 
# There are various stemmers and one lemmatizer in NLTK, the most common being:
# 
#  - **Porter Stemmer** from [Porter (1980)](https://tartarus.org/martin/PorterStemmer/index.html)
#  - **Wordnet Lemmatizer** (port of the Morphy: https://wordnet.princeton.edu/man/morphy.7WN.html)

# In[ ]:


from nltk.stem import PorterStemmer
porter = PorterStemmer()

for word in ['walking', 'walks', 'walked']:
    print(porter.stem(word))


# In[ ]:


from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()

for word in ['walking', 'walks', 'walked']:
    print(wnl.lemmatize(word))


# # Gotcha! The lemmatizer is actually pretty complicated, it needs Parts of Speech (POS) tags.
# 
# 
# We won't cover what's POS today so I'll just show you how to "whip" the lemmatizer to do what you need.
# 
# By default, the WordNetLemmatizer.lemmatize() function will assume that the word is a Noun if there's no explict POS tag in the input.
# 
# First you need the pos_tag function to tag a sentence and using the tag convert it into WordNet tagsets and then put it through to the WordNetLemmatizer.
# 
# **Note:** Lemmatization won't really work on single words alone without context or knowledge of its POS tag (i.e. we need to know whether the word is a noun, verb, adjective, adverb)
# 

# In[ ]:


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' # if mapping isn't found, fall back to Noun.
    
# `pos_tag` takes the tokenized sentence as input, i.e. list of string,
# and returns a tuple of (word, tg), i.e. list of tuples of strings
# so we need to get the tag from the 2nd element.

walking_tagged = pos_tag(word_tokenize('He is walking to school'))
print(walking_tagged)


# In[ ]:


[wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) for word, tag in walking_tagged]


# # Now, lets create a new lemmatization function for sentences given what we learnt above.

# In[ ]:


from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

wnl = WordNetLemmatizer()

def penn2morphy(penntag):
    """ Converts Penn Treebank tags to WordNet. """
    morphy_tag = {'NN':'n', 'JJ':'a',
                  'VB':'v', 'RB':'r'}
    try:
        return morphy_tag[penntag[:2]]
    except:
        return 'n' 
    
def lemmatize_sent(text): 
    # Text input is string, returns lowercased strings.
    return [wnl.lemmatize(word.lower(), pos=penn2morphy(tag)) 
            for word, tag in pos_tag(word_tokenize(text))]

lemmatize_sent('He is walking to school')


# # Lets try the `lemmatize_sent()` and remove stopwords from Single no. 8

# In[ ]:


print('Original Single no. 8:')
print(single_no8, '\n')
print('Lemmatized and removed stopwords:')
print([word for word in lemmatize_sent(single_no8) 
       if word not in stoplist_combined
       and not word.isdigit() ])


# # Combining what we know about removing stopwords and lemmatization

# In[ ]:


def preprocess_text(text):
    # Input: str, i.e. document/sentence
    # Output: list(str) , i.e. list of lemmas
    return [word for word in lemmatize_sent(text) 
            if word not in stoplist_combined
            and not word.isdigit()]


# # Tangential Note on Lemmatization
# 
# In English, a root word / lemma can manifest in different forms. 
# 
# | <img src="https://media1.giphy.com/media/xHHsXH7WJsWK4/giphy.gif" align="left" height="200" width="200"> | <img src="https://media1.giphy.com/media/xHHsXH7WJsWK4/giphy.gif" align="left" height="200" width="200"><img src="https://media1.giphy.com/media/xHHsXH7WJsWK4/giphy.gif" align="left" height="200" width="200"> |
# |:-------------:|:-------------:|
# | 1 cat  | 2 cats  |
# | 1 cat  | 2 cats  |
# 
# For instance, we use “cat” to refer to a single “cat” and we attach an “-s”  suffix to refer to more than one cat, e.g. “two cats”. 
# 
# | <img src="https://68.media.tumblr.com/b0755247c8f32f79413d34b0410ccff1/tumblr_o3q8wlGi9v1u9ia8fo1_500.gif" align="left" height="200" width="400"> | 
# |:-------------:| 
# | cats walk / cats (are) walking | 
# 
# <!-- | <img src="https://upload.wikimedia.org/wikipedia/commons/thumb/f/fb/ModelsCatwalk.jpg/440px-ModelsCatwalk.jpg" align="left" height="200" width="400"> | 
# |:-------------:| 
# | cat walk(s) / catwalk(s) |  --> 
# 
# 
# Another example, the word “walk” has different forms, e.g. “walking” and “walked” indicate the time and/or progress of the walking motion. <!-- ~~Additionally, “walk” can also refer to the act of walking which is different from the walking motion, e.g. "John went for a walk" (act of walking) vs "John wanted to walk to the park" (the walking action/motion).~~ --> We refer to these root words as ***word types*** (e.g. “cat” and “walk”) and their different forms as ***word tokens*** (e.g. “cats”, “walk”, “walking”, “walked”, “walks”). 
# 
# Linguists further distinguish words between their lemmas or word families. A lemma refers to the canonical root word used as a dictionary entry. A word family refers to a group of lemmas which are derived from a single root word. Even though "walkable" would be a separate entry in a dictionary from "walk", "walkable" can be grouped under the word family of "walk" together with "walking, walked, walks".
# 
# The distinction is subtle yet linguists go into great length to argue for what counts as a type, token, lemmas or word family. 

# # From Strings to Vectors
# 
# **Vector** is an array of numbers
# 
# **Vector Space Model** is conceptualizing language as a whole lot of numbers
# 
# **Bag-of-Words (BoW)**: Counting each document/sentence as a vector of numbers, with each number representing the count of a word in the corpus
# 
# To count, we can use the Python `collections.Counter`

# In[ ]:


from collections import Counter

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

# Lemmatize and remove stopwords
processed_sent1 = preprocess_text(sent1)
processed_sent2 = preprocess_text(sent2)


# In[ ]:


print('Processed sentence:')
print(processed_sent1)
print()
print('Word counts:')
print(Counter(processed_sent1))


# In[ ]:


print('Processed sentence:')
print(processed_sent2)
print()
print('Word counts:')
print(Counter(processed_sent2))


# # Vectorization
# 
# Let's put the words and counts into a nice table:
# 
# | | brown | quick | fox | jump | lazy | dog | mr | 
# |:---- |:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|:-----:|
# | Sent1 | 2 | 1 | 1 | 1 | 1 | 1 | 0 |  
# | Sent2 | 1 | 0 | 1 | 1 | 1 | 0 | 1 | 
# 
# 
# If we fix the positions of the vocabulary i.e. 
# 
# ```
# [brown, quick, fox, jump, lazy, dog, mr]
# ```
# 
# and we do the counts for each word in each sentence, we get the sentence vectors (i.e. list of numbers to represent each sentence):
# 
# ```
# sent1 = [2,1,1,1,1,1,0]
# sent2 = [1,0,1,1,1,0,1]
# ```

# # Vectorization with sklearn 
# 
# In `scikit-learn`, there're pre-built functions to do the preprocessing and vectorization that we've been doing using the `CountVectorizer` object. 
# 
# It will be the object that contains the vocabulary (i.e. the first row of our table above) and has the function to convert any sentence into the counts vectors we see as above.
# 
# The input that `CountVectorizer` is a textfile, so we've to do some hacking to put let it accept the string outputs.
# 
# We can "fake it to make it" using `io.StringIO` where we can convert any string to work like a file, e.g. 

# In[ ]:


from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

with StringIO('\n'.join([sent1, sent2])) as fin:
    # Create the vectorizer
    count_vect = CountVectorizer()
    count_vect.fit_transform(fin)


# In[ ]:


# We can check the vocabulary in our vectorizer
# It's a dictionary where the words are the keys and 
# The values are the IDs given to each word. 
count_vect.vocabulary_


# **Note:** We haven't counted anything yet just initializing our vectorizer object with the vocabulary. 

# # ちょっと待ってください ... (Wait a minute)
# 
# I didn't tell the vectorizer to remove punctuation and tokenize and lowercase, how did they do it?
# 
# Also, `the` is in the vocabulary, it's a stopword, we want it gone... <br>
# And `jumps` isn't stemmed or lemmatized!

# If we look at the documentation of the [`CountVectorizer`](http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) in `sklearn`, we see:
# 
# 
# ```python
# CountVectorizer(
#     input=’content’, encoding=’utf-8’, 
#     decode_error=’strict’, strip_accents=None, 
#     lowercase=True, preprocessor=None, 
#     tokenizer=None, stop_words=None, 
#     token_pattern=’(?u)\b\w\w+\b’, ngram_range=(1, 1), 
#     analyzer=’word’, max_df=1.0, min_df=1, 
#     max_features=None, vocabulary=None, 
#     binary=False, dtype=<class ‘numpy.int64’>)[source]
# ```
# 
# And more specifically:
# 
# > **analyzer** : string, {‘word’, ‘char’, ‘char_wb’} or callable
# > 
# > Whether the feature should be made of word or character n-grams. Option ‘char_wb’ creates character n-grams only from text inside word boundaries; n-grams at the edges of words are padded with space.
# > If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.
# 
#  
# > **preprocessor** : callable or None (default)
# > 
# > Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams generation steps.
# 
# > **tokenizer** : callable or None (default)
# > 
# > Override the string tokenization step while preserving the preprocessing and n-grams generation steps. Only applies if analyzer == 'word'.
# 
# > **stop_words** : string {‘english’}, list, or None (default)
# > 
# > If ‘english’, a built-in stop word list for English is used.
# > If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens. Only applies if analyzer == 'word'.
# If None, no stop words will be used. 
# 
# > **lowercase** : boolean, True by default
# > 
# > Convert all characters to lowercase before tokenizing.

# # Achso, we can override these arguments with the functions we have learnt before.

# We can **override the tokenizer and stop_words**:

# In[ ]:


from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

with StringIO('\n'.join([sent1, sent2])) as fin:
    # Override the analyzer totally with our preprocess text
    count_vect = CountVectorizer(stop_words=stoplist_combined,
                                 tokenizer=word_tokenize)
    count_vect.fit_transform(fin)
count_vect.vocabulary_


# Or just **override the analyzer** totally with our preprocess text:

# In[ ]:


from io import StringIO
from sklearn.feature_extraction.text import CountVectorizer

sent1 = "The quick brown fox jumps over the lazy brown dog."
sent2 = "Mr brown jumps over the lazy fox."

with StringIO('\n'.join([sent1, sent2])) as fin:
    # Override the analyzer totally with our preprocess text
    count_vect = CountVectorizer(analyzer=preprocess_text)
    count_vect.fit_transform(fin)
count_vect.vocabulary_ 


# # To vectorize any new sentences, we use  `CountVectorizer.transform()` 
# 
# The function  will return a sparse matrix.

# In[ ]:


count_vect.transform([sent1, sent2])


# # To view the matrix, you can output it to an array

# In[ ]:


from operator import itemgetter

# Print the words sorted by their index
words_sorted_by_index, _ = zip(*sorted(count_vect.vocabulary_.items(), key=itemgetter(1)))

print(preprocess_text(sent1))
print(preprocess_text(sent2))
print()
print('Vocab:', words_sorted_by_index)
print()
print('Matrix/Vectors:\n', count_vect.transform([sent1, sent2]).toarray())


# Naive Bayes 
# ====
# 
# 
# 
# 
# Classification
# ====
# 
# Classification simply means putting our data points into bins/box. You can also think of it as assigning label to our data points, e.g. given box of fruits, sort them in apples, oranges and others. 
# 
# Okay, the explanation could be more complex than that but `import this` says:
# 
# > **Simple is better than complex.**

# # Now that we learnt some basic NLP and vectorization, lets apply it to a fun task.

# [Random Acts of Pizza](https://www.kaggle.com/c/random-acts-of-pizza)
# =====
# 
# In machine learning, it is often said there are [no free lunches](). How wrong we were.
# 
# This competition contains a dataset with 5671 textual requests for pizza from the Reddit community Random Acts of Pizza together with their outcome (successful/unsuccessful) and meta-data. 
# 
# ![](https://kaggle2.blob.core.windows.net/competitions/kaggle/3949/media/pizzas.png)
# 
# The task is to create an algorithm capable of predicting which requests will garner a cheesy (but sincere!) act of kindness.
# 

# # Lets take a look at the training data

# In[ ]:


import json

with open('../input/random-acts-of-pizza/train.json') as fin:
    trainjson = json.load(fin)


# In[ ]:


trainjson[0]


# We're only interested in the text fields:
# 
# **Input**:
#  - `request_id`: unique identifier for the request 
#  - `request_title`: title of the reddit post for pizza request
#  - `request_text_edit_aware`: expository to request for pizza
#  
# **Output**:
#  - `requester_recieved_pizza`: whether requester gets his/her pizza
#  
# For our purpose, lets only use the `request_text` as the input to build our Naive Bayes classifier and the output is the `requester_recieved_pizza` field.
# 
# **Note:** The `request_id` is only used for mapping purpose when we're submitting the results to the Kaggle task.

# In[ ]:


print('UID:\t', trainjson[0]['request_id'], '\n')
print('Title:\t', trainjson[0]['request_title'], '\n')
print('Text:\t', trainjson[0]['request_text_edit_aware'], '\n')
print('Tag:\t', trainjson[0]['requester_received_pizza'], end='\n')


# # Here's a neat trick to convert json to pandas DataFrame

# In[ ]:


import pandas as pd
df = pd.io.json.json_normalize(trainjson) # Pandas magic... 
df_train = df[['request_id', 'request_title', 
               'request_text_edit_aware', 
               'requester_received_pizza']]
df_train.head()


# # Lets take a look at the test data

# In[ ]:


import json

with open('../input/random-acts-of-pizza/test.json') as fin:
    testjson = json.load(fin)


# In[ ]:


print('UID:\t', testjson[0]['request_id'], '\n')
print('Title:\t', testjson[0]['request_title'], '\n')
print('Text:\t', testjson[0]['request_text_edit_aware'], '\n')
print('Tag:\t', testjson[0]['requester_received_pizza'], end='\n')


# # Gotcha again! 
# 
# In the test data, our label (i.e. `requester_received_pizza`) **won't be known** to us since that's the thing that our classifier is predicting.
# 
# **Note:** Whatever features that we're going to train our classifier with, we should have them in our test set too. In our case we need to make sure that the test set has `request_text_edit_aware` field.

# # Lets put the test data into a pandas DataFrame too

# In[ ]:


import pandas as pd
df = pd.io.json.json_normalize(testjson) # Pandas magic... 
df_test = df[['request_id', 'request_title', 
               'request_text_edit_aware']]
df_test.head()


# # Split training data before vectorization
# 
# The first thing to do is to split our training data into 2 parts:
# 
#  - **training**: Use for training our model
#  - **validation**: Use to check the "soundness" of our model
#  
# **Note:** 
# 
#  - Splitting the data into 2 parts and holding out one part to check the model is one of method to validate the "soundness" of our model. It's call the **hold-out** validation. 
# 
#  - Another popular validation method is **cross-validation**, it's out of scope here but you can take a look at `crossvalidation` in `scikit-learn`. 
#  

# In[ ]:


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split 

# It doesn't really matter what the function name is called
# but the `train_test_split` is splitting up the data into 
# 2 parts according to the `test_size` argument you've set.

# When we're splitting up the training data, we're spltting up 
# into train, valid split. The function name is just a name =)
train, valid = train_test_split(df_train, test_size=0.2)


# # Vectorize the train and validation set

# In[ ]:


# Initialize the vectorizer and 
# override the analyzer totally with the preprocess_text().
# Note: the vectorizer is just an 'empty' object now.
count_vect = CountVectorizer(analyzer=preprocess_text)

# When we use `CounterVectorizer.fit_transform`,
# we essentially create the dictionary and 
# vectorize our input text at the same time.
train_set = count_vect.fit_transform(train['request_text_edit_aware'])
train_tags = train['requester_received_pizza']

# When vectorizing the validation data, we use `CountVectorizer.transform()`.
valid_set = count_vect.transform(valid['request_text_edit_aware'])
valid_tags = valid['requester_received_pizza']


# # Now, we need to vectorize the test data too
# 
# After we vectorize our data, the input to train the classifier would be the vectorized text. 
# <br>When we predict the label with the trained mdoel, our input needs to be vectorized too.
# 

# In[ ]:


# When vectorizing the test data, we use `CountVectorizer.transform()`.
test_set = count_vect.transform(df_test['request_text_edit_aware'])


# # Naive Bayes classifier in sklearn
# 
# There are different variants of Naive Bayes (NB) classifier in `sklearn`. <br>
# For simplicity, lets just use the `MultinomialNB`.
# 
# **Multinomial** is a big word but it just means many classes/categories/bins/boxes that needs to be classified. 

# In[ ]:


from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB() 

# To train the classifier, simple do 
clf.fit(train_set, train_tags) 


# # Before we test our classifier on the test set, we get a sense of how good it is on the validation set.

# In[ ]:


from sklearn.metrics import accuracy_score

# To predict our tags (i.e. whether requesters get their pizza), 
# we feed the vectorized `test_set` to .predict()
predictions_valid = clf.predict(valid_set)

print('Pizza reception accuracy = {}'.format(
        accuracy_score(predictions_valid, valid_tags) * 100)
     )


# # Now lets use the full training data set and re-vectorize and retrain the classifier
# 
# More data == better model (in most cases)

# In[ ]:


count_vect = CountVectorizer(analyzer=preprocess_text)

full_train_set = count_vect.fit_transform(df_train['request_text_edit_aware'])
full_tags = df_train['requester_received_pizza']

# Note: We have to re-vectorize the test set since
#       now our vectorizer is different using the full 
#       training set.
test_set = count_vect.transform(df_test['request_text_edit_aware'])

# To train the classifier
clf = MultinomialNB() 
clf.fit(full_train_set, full_tags) 


# # Finally, we use the classifier to predict on the test set

# In[ ]:


# To predict our tags (i.e. whether requesters get their pizza), 
# we feed the vectorized `test_set` to .predict()
predictions = clf.predict(test_set)


# **Note:** Since we don't have the `requester_received_pizza` field in test data, we can't measure accuracy. But we can do some exploration as shown below.

# # From the training data, we had 24% pizza giving rate

# In[ ]:


success_rate = sum(df_train['requester_received_pizza']) / len(df_train) * 100
print(str('Of {} requests, only {} gets their pizzas,'
          ' {}% success rate...'.format(len(df_train), 
                                        sum(df_train['requester_received_pizza']), 
                                       success_rate)
         )
     )


# # Lolz, our classifier is rather stingy...

# In[ ]:


success_rate = sum(predictions) / len(predictions) * 100
print(str('Of {} requests, only {} gets their pizzas,'
          ' {}% success rate...'.format(len(predictions), 
                                        sum(predictions), 
                                       success_rate)
         )
     )


# # How accurate is our count vectorization naive bayes classifier on the test data?
# 
# Since we don't have the `requester_received_pizza` field in the test data, we have to check that with an oracle (i.e. the person that knows). 
# 
# On Kaggle, **checking with the oracle** means uploading the file in the correct format and their script will process the scores and tell you how you did.
# 
# **Note:** Different tasks will use different metrics but in most cases getting as many correct predictions as possible is the thing to aim for. We won't get into the details of how classifiers are evaluated but for a start, please see [precision, recall and F1-scores](https://en.wikipedia.org/wiki/Precision_and_recall) 
# 

# # Finally, lets take a look at what format the oracle expects and create the output file for our predictions accordingly

# In[ ]:


df_sample_submission = pd.read_csv('../input/patching-pizzas/sampleSubmission.csv')
df_sample_submission.head()


# In[ ]:


# We've kept the `request_id` previous in the `df_test` dataframe.
# We can simply merge that column with our predictions.
df_output = pd.DataFrame({'request_id': list(df_test['request_id']), 
                          'requester_received_pizza': list(predictions)}
                        )
# Convert the predictions from boolean to integer.
df_output['requester_received_pizza'] = df_output['requester_received_pizza'].astype(int)
df_output.head()


# In[ ]:


# Create the csv file.
df_output.to_csv('basic-nlp-submission.csv')

