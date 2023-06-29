import nltk
import spacy
import numpy as np
from nltk.corpus import stopwords
from gensim.utils import simple_preprocess
from nltk.tokenize import sent_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('stopwords')

stops = set(stopwords.words("english"))

def lemmatize(texts: str, allowed_post_tags=["NOUN", "ADJ", "VERB", "ADV", "PROPN"], accuracy = "low") :
    """
    :param texts: paragraphs
    :param allowed_post_tags: allowed parts of speech
    :param accuracy: (optional) accuracy needed from the model 
        return: 
            tokenized and lemmatized texts
    """
    if type(texts) != str: 
        raise Exception("No input string given.")
    
    if accuracy not in ["low", "high"]:
        raise Exception("In correct argument:", accuracy)
    else:
        if accuracy != "high":
            #Use "python -m spacy download en_core_web_sm" in a terminal if error [E050]
            model = spacy.load("en_core_web_sm", disable=["parser","ner"])
        else:
            #Use "python -m spacy download 'en_core_web_trf'" in a terminal if error [E050]
            model = spacy.load('en_core_web_md', disable=["parser","ner"])

    model.add_pipe('sentencizer')
    sentence_generator = model(texts).sents # create a generator object for sentences in the text
    sentence_arr = []

    for sentence in sentence_generator:
        tokens = " ".join([token.lemma_ for token in sentence
                        if token.pos_ in allowed_post_tags 
                        and token not in stops
                        and not token.is_punct
                        and not token.like_num
                        and not token.is_digit
                        and not token.is_space
                        and not token.is_currency]) # checking all this takes a lot of time
        
        sentence_arr.append(tokens)
        
    return " ".join(sentence_arr)

def gen_words(tokens: str):
    tokens = " ".join(simple_preprocess(tokens, deacc=True))
    
    return (tokens)

def main(): 
    # test
    paragraph = f"In the tranquil meadows of a forgotten countryside, where time seemed to stretch its arms lazily across the horizon, a gentle breeze whispered secrets to the tall grass, swaying it in rhythmic undulations. The sun, ablaze with golden hues, cast its radiant beams upon the idyllic landscape, illuminating every blade of grass and infusing the air with a warm embrace. As the day unfolded, birds soared through the vast expanse of the sky, their wings outstretched in graceful arcs, painting fleeting patterns against the canvas of the heavens. Amidst this picturesque scene, a solitary figure, clad in a flowing cloak of vibrant colors, stood atop a hill, gazing at the panoramic vista that lay before them."
        
    print(lemmatize(paragraph,accuracy= "high"))

if __name__ == "__main__": 
    main()