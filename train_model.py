__author__ = 'Jmexe'
import gensim
from nltk.corpus import stopwords
import nltk

class MySentences(object):
    """
    Sentence object
    """
    def __init__(self, filename):
        self.filename = filename
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


def train_your_own(text, model):
    """
    Train model using your own text
    :param text:
    :param model:
    :return:
    """

    #Split the text into sentences
    sentences_raw = nltk.sent_tokenize(text)

    #prepare stop words
    stop_words = stopwords.words('english')
    stop_types = [',', 'POS', '``', "''", '.', "''", '?', '!']
    stemmer = nltk.stem.porter.PorterStemmer()

    #cut the sentences into words and remove all stop words
    sentences = []
    for sentence in sentences_raw:
        tokens = nltk.pos_tag(nltk.word_tokenize(sentence))

        #Remove stop words and trim the words
        good_words = []
        for w,wtype in tokens:
            if wtype not in stop_types and w not in stop_words:
                good_words.append(stemmer.stem(w))
        sentences.append(good_words)

    if model == None:
        model =gensim.models.Word2Vec(sentences, min_count=1)
    else:
        model.train(sentences)

    return model



def train_wiki(file_path):
    """
    Train a model using First billion characters from wikipedia
    See reference here : http://mattmahoney.net/dc/textdata.html
    """
    sentences = MySentences(file_path)
    model = gensim.models.Word2Vec(sentences)

    model.save("./wiki9")
    return model


if __name__ == '__main__':
    text = "Police say a wrong-way driver struck the front of a construction truck and went airborne, causing five injuries in Lowell." \
           "According to Lowell Police, a 76-year-old Lowell woman drove the wrong way down Appleton Street into a construction site. Her husband was riding in the car with her." \
           "After speeding around a traffic jam, her car crashed into the back of a New England Utility Constructors Inc. Bobcat machine. The Bobcat spun around into a construction worker, launching him 30 feet into the air." \
           "Police say the impact sent the car into the air as well. It landed in the back of one of the construction trucks." \
           "A construction worker injured his leg when he leapt out of the way of the flying vehicle. He was taken to Lowell General Hospital for treatment." \
           "The worker who was struck by the Bobcat has also been taken to hospital for serious injuries, but he is currently stable. The operator of the Bobcat was taken to Lowell General as well." \
           "Police say that the woman and her husband were also injured and transported to the hospital, but she refused treatment." \
           "The investigation is ongoing, and it is unknown if charges will be pressed."

    model = train_your_own(text, None)

    print model["worker"]
