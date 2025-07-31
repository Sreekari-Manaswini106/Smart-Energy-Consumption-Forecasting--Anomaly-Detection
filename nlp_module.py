from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json

class NLPEnergyAssistant:
    def __init__(self, faq_file='faq_data.json'):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        with open(faq_file, 'r', encoding='utf-8') as file:
            self.data = json.load(file)
        self.questions = [item['question'] for item in self.data]
        self.answers = [item['answer'] for item in self.data]
        self.question_embeddings = self.model.encode(self.questions)

    def get_response(self, user_input):
        input_embedding = self.model.encode([user_input])
        similarities = cosine_similarity(input_embedding, self.question_embeddings)
        best_idx = np.argmax(similarities)
        score = similarities[0][best_idx]

        if score > 0.6:
            return self.answers[best_idx]
        else:
            return "I'm not sure how to answer that. Please try rephrasing your question or ask something else."
