import os
from sklearn.metrics.pairwise import pairwise_distances_argmin
from utils import *
import random
from urllib.request import urlopen
from bs4 import BeautifulSoup

class ThreadRanker(object):
    def __init__(self, paths):
        self.word_embeddings, self.embeddings_dim = load_embeddings(paths['WORD_EMBEDDINGS'])
        self.thread_embeddings_folder = paths['THREAD_EMBEDDINGS_FOLDER']

    def __load_embeddings_by_tag(self, tag_name):
        embeddings_path = os.path.join(self.thread_embeddings_folder, tag_name + ".pkl")
        thread_ids, thread_embeddings = unpickle_file(embeddings_path)
        return thread_ids, thread_embeddings

    def get_best_thread(self, question, tag_name):
        thread_ids, thread_embeddings = self.__load_embeddings_by_tag(tag_name)
        
        question_vec = question_to_vec(question, self.word_embeddings, self.embeddings_dim)
        best_thread = pairwise_distances_argmin(
            X=question_vec.reshape(1, self.embeddings_dim),
            Y=thread_embeddings,
            metric='cosine'
        )
        
        return thread_ids[best_thread][0]


class DialogueManager(object):
    def __init__(self, paths):
        print("Loading resources...")

        # Intent recognition:
        self.intent_recognizer = unpickle_file(paths['INTENT_RECOGNIZER'])
        self.tfidf_vectorizer = unpickle_file(paths['TFIDF_VECTORIZER'])

        self.ANSWER_TEMPLATE = 'https://stackoverflow.com/questions/%s'

        # Goal-oriented part:
        self.tag_classifier = unpickle_file(paths['TAG_CLASSIFIER'])
        self.thread_ranker = ThreadRanker(paths)

    def scrapper(self, url):
        response = urlopen(url)
        html = response.read()
        soup = BeautifulSoup(html,'lxml')
        answerSet=soup.body.find_all("div", class_="answercell post-layout--right")
        answer = answerSet[0].find_all("div",class_="post-text")
        for tag in answer:
          return tag.text


    # def replace_url_to_link(self,value):
    #     urls = re.compile(r"((https?):((//)|(\\\\))+[\w\d:#@%/;$()~_?\+-=\\\.&]*)", re.MULTILINE|re.UNICODE)
    #     value = urls.sub(r"\1", value)
    #     return value


    def create_chitchat_bot(self,sentence):
        
        GREETING_KEYWORDS = ("hello", "hi", "greetings", "sup","hey")

        GREETING_RESPONSES = ( "Hey","Hi","Greetings","How can i help you?" )

        for word in sentence.split(" "):
            if word.lower() in GREETING_KEYWORDS:
                return random.choice(GREETING_RESPONSES)


       
    def generate_answer(self, question):
        
        prepared_question = text_prepare(question)
        features = self.tfidf_vectorizer.transform([prepared_question])
        intent = self.intent_recognizer.predict(features)[0]

        # Chit-chat part:   
        if intent == 'dialogue':
            response = self.create_chitchat_bot(question)
            return response
        
        # Goal-oriented part:
        else:        
            tag = self.tag_classifier.predict(features)[0]
            
            comb=()
            thread_id = self.thread_ranker.get_best_thread(prepared_question, tag)
           
            x = self.scrapper(self.ANSWER_TEMPLATE % (thread_id))
            # y = self.replace_url_to_link(self.ANSWER_TEMPLATE % (thread_id))
            y = self.ANSWER_TEMPLATE % (thread_id)
            return x+"||"+ y
            # return comb
