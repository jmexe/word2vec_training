__author__ = 'Jmexe'
from prepare_data import prepare
import gensim
from nltk.corpus import stopwords
import nltk
import numpy as np

def train():
    text = "Police say a wrong-way driver struck the front of a construction truck and went airborne, causing five injuries in Lowell." \
           "According to Lowell Police, a 76-year-old Lowell woman drove the wrong way down Appleton Street into a construction site. Her husband was riding in the car with her." \
           "After speeding around a traffic jam, her car crashed into the back of a New England Utility Constructors Inc. Bobcat machine. The Bobcat spun around into a construction worker, launching him 30 feet into the air." \
           "Police say the impact sent the car into the air as well. It landed in the back of one of the construction trucks." \
           "A construction worker injured his leg when he leapt out of the way of the flying vehicle. He was taken to Lowell General Hospital for treatment." \
           "The worker who was struck by the Bobcat has also been taken to hospital for serious injuries, but he is currently stable. The operator of the Bobcat was taken to Lowell General as well." \
           "Police say that the woman and her husband were also injured and transported to the hospital, but she refused treatment." \
           "The investigation is ongoing, and it is unknown if charges will be pressed."


    sents = nltk.sent_tokenize(text)

    stop_words = stopwords.words('english')
    stop_types = [',', 'POS', '``', "''", '.', "''", '?', '!']
    stemmer = nltk.stem.porter.PorterStemmer()

    arr_text = []
    for sentence in sents:
        tokens = nltk.pos_tag(nltk.word_tokenize(sentence))

        #Remove stop words and trim the words
        good_words = []
        for w,wtype in tokens:
            if wtype not in stop_types and w not in stop_words:
                good_words.append(stemmer.stem(w))
        arr_text.append(good_words)

    model= gensim.models.Word2Vec(arr_text, min_count=1)

    model.save("./polmodel")

class MySentences(object):
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()

def train_wiki():
    sentences = MySentences('/Users/Jmexe/Develop/Data/word2vec/training_text')
    model = gensim.models.Word2Vec(sentences)

    model.save("./wiki9")



if __name__ == '__main__':

    model = gensim.models.Word2Vec.load("./wiki9")
    print model["hello"]

    """
    model = gensim.models.Word2Vec.load("./polmodel")

    text = "Harris tied a career best with four, three-point field goals in the outing. Redshirt freshman Dontavious Smith (Cullen, La.) paced the team on the boards, pulling down a personal-best eight rebounds. Junior Tyler Livingston added to the offense with nine points for the third consecutive game, while redshirt sophomore Jahad Thomas (Williamsport, Pa.) posted seven points and seven rebounds, as well."

    sents = nltk.sent_tokenize(text)

    stop_words = stopwords.words('english')
    stop_types = [',', 'POS', '``', "''", '.', "''", '?', '!']
    stemmer = nltk.stem.porter.PorterStemmer()

    arr_text = []
    for sentence in sents:
        tokens = nltk.pos_tag(nltk.word_tokenize(sentence))

        #Remove stop words and trim the words
        good_words = []
        for w,wtype in tokens:
            if wtype not in stop_types and w not in stop_words:
                good_words.append(stemmer.stem(w))
        arr_text.append(good_words)

    model.train(arr_text)
    """