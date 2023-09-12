import contractions
import re
import nltk
from nltk.corpus import stopwords
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences

nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
# initializing Stop words libraries
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))

def clean_text(text, noisy_words_path):
    desc = contractions.fix(text)
    # for word in noisy_words:
    #     desc = re.sub(r'\b' + word + r'\b', '', desc)

    # Open the text file and read the lines into a list
    with open(noisy_words_path, 'r') as f:
        words = f.readlines()

    # Remove any newline characters from each word
    noisy_words = [word.strip() for word in words]

    desc = re.sub("[!@.$\'\'':()]", "", desc)
    tokens = nltk.word_tokenize(desc)
    cleaned_tokens = [word for word in tokens if word not in noisy_words]
    cleaned_description = ' '.join(cleaned_tokens)
    return cleaned_description

def tokenize_and_tag(desc):
    tokens = nltk.word_tokenize(desc.lower())
    filtered_tokens = [w for w in tokens if not w in stop_words]
    tagged = nltk.pos_tag(filtered_tokens)
    return tagged

def extract_POS(tagged):
    #pattern 1
    grammar1 = ('''Noun Phrases: {<DT>?<JJ>*<NN|NNS|NNP>+}''')
    chunkParser = nltk.RegexpParser(grammar1)
    tree1 = chunkParser.parse(tagged)

    # typical noun phrase pattern appending to be concatted later
    g1_chunks = []
    for subtree in tree1.subtrees(filter=lambda t: t.label() == 'Noun Phrases'):
        g1_chunks.append(subtree)

    #pattern 2
    grammar2 = ('''NP2: {<IN>?<JJ|NN>*<NNS|NN>} ''')
    chunkParser = nltk.RegexpParser(grammar2)
    tree2 = chunkParser.parse(tagged)

    # variation of a noun phrase pattern to be pickled for later analyses
    g2_chunks = []
    for subtree in tree2.subtrees(filter=lambda t: t.label() == 'NP2'):
        g2_chunks.append(subtree)

    #pattern 3
    grammar3 = (''' VS: {<VBG|VBZ|VBP|VBD|VB|VBN><NNS|NN>*}''')
    chunkParser = nltk.RegexpParser(grammar3)
    tree3 = chunkParser.parse(tagged)

    # verb-noun pattern appending to be concatted later
    g3_chunks = []
    for subtree in tree3.subtrees(filter=lambda t: t.label() == 'VS'):
        g3_chunks.append(subtree)


    # pattern 4
    # any number of a singular or plural noun followed by a comma followed by the same noun, noun, noun pattern
    grammar4 = ('''Commas: {<NN|NNS>*<,><NN|NNS>*<,><NN|NNS>*} ''')
    chunkParser = nltk.RegexpParser(grammar4)
    tree4 = chunkParser.parse(tagged)

    # common pattern of listing skills appending to be concatted later
    g4_chunks = []
    for subtree in tree4.subtrees(filter=lambda t: t.label() == 'Commas'):
        g4_chunks.append(subtree)

    return g1_chunks, g2_chunks, g3_chunks, g4_chunks

def training_set(chunks):
    '''creates a dataframe that easily parsed with the chunks data '''
    df = pd.DataFrame(chunks)
    df.fillna('X', inplace = True)

    train = []
    for row in df.values:
        phrase = ''
        for tup in row:
            # needs a space at the end for seperation
            phrase += tup[0] + ' '
        phrase = ''.join(phrase)
        # could use padding tages but encoder method will provide during
        # tokenizing/embeddings; X can replace paddding for now
        train.append( phrase.replace('X', '').strip())

    df['phrase'] = train

    #returns 50% of each dataframe to be used if you want to improve execution time
    # return df.phrase.sample(frac = 0.5)
    # Update: only do 50% if running on excel
    return df.phrase

def strip_commas(df):
    '''create new series of individual n-grams'''
    grams = []
    for sen in df:
        sent = sen.split(',')
        for word in sent:
            grams.append(word)
    return pd.Series(grams)

def generate_phrases(desc):
    desc = clean_text(desc)
    tagged = tokenize_and_tag(desc)
    g1_chunks, g2_chunks, g3_chunks, g4_chunks = extract_POS(tagged)
    c = training_set(g4_chunks)
    separated_chunks4 = strip_commas(c)
    phrases = pd.concat([training_set(g1_chunks),
                          training_set(g2_chunks),
                          training_set(g3_chunks),
                          separated_chunks4],
                            ignore_index = True )
    return phrases



def extract_skills_using_lstm(trained_lstm_model, raw_job_description, tokenizer, max_length, noisy_words_path = 'noisy_words.txt'):
    # Tokenize and tag the new job description
    tagged_description = tokenize_and_tag(raw_job_description)

    # Extract parts of speech
    g1_chunks, g2_chunks, g3_chunks, g4_chunks = extract_POS(tagged_description)

    c = training_set(g4_chunks)
    separated_chunks4 = strip_commas(c)
    phrases = pd.concat([training_set(g1_chunks),
                        training_set(g2_chunks),
                        training_set(g3_chunks),
                        separated_chunks4],
                            ignore_index = True)
    
    # Open the text file and read the lines into a list
    with open(noisy_words_path, 'r') as f:
        words = f.readlines()

    # Remove any newline characters from each word
    noisy_words = [word.strip() for word in words]
    
    noisy_words = list(set(noisy_words))

    def preprocess_phrases(text):
        # remove leading and trailing spaces
        text = text.strip()
        # remove leading and trailing slashes
        text = text.strip('/')
        # remove noisy words
        text = ' '.join(word for word in text.split() if word not in noisy_words)
        return text
    
    # Keep rows that are not in the noisy_words list
    filtered_phrases = phrases[~phrases.isin(noisy_words)]

    # Keep rows that don't contain special characters except '-', '/', and '.'
    filtered_phrases = filtered_phrases[filtered_phrases.apply(lambda x: bool(re.match('^[a-zA-Z-/\. ]*$', x)))]

    filtered_phrases = filtered_phrases.apply(lambda row: preprocess_phrases(row))

    # Remove empty strings
    filtered_phrases = filtered_phrases[filtered_phrases != '']

    # Remove duplicates
    filtered_phrases = filtered_phrases.drop_duplicates()

    # Keep only phrases with at most 3 words
    filtered_phrases = filtered_phrases[filtered_phrases.apply(lambda x: len(x.split(' ')) <= 3)]

    # create array of text
    text = np.array(filtered_phrases)

    # convert the array of text into sequence
    encoded_docs = tokenizer.texts_to_sequences(text)

    # pad the text sequence to make them equal length
    padded_docs = pad_sequences(encoded_docs, maxlen=max_length, padding='post')

    # perform the prediction using trained lstm model
    predictions = (trained_lstm_model.predict(padded_docs) > 0.5).astype('int32')

    # perform the prediction using lstm
    out = pd.DataFrame({'Phrase':filtered_phrases, 'Class':predictions.ravel()})

    # extract the skill from the predicted classification of phrases
    skills = out.loc[out['Class'] == 1]

    return skills['Phrase'].tolist()
