from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from IPython.display import display, HTML


class LDA:
    def __init__(self, n_components=10):
        self.n_components = n_components
        self.vectorizer = CountVectorizer()
        self.lda_model = None
        self.feature_names = None

    def fit(self, documents):
        # Convert text into a matrix of token counts
        X = self.vectorizer.fit_transform(documents)

        # Apply Latent Dirichlet Allocation
        self.lda_model = LatentDirichletAllocation(
            n_components=self.n_components, random_state=42
        )
        self.lda_model.fit(X)

        # Get the feature names
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_topics(self, num_top_words=5):
        if self.lda_model is None:
            raise Exception("LDA model not fitted.")

        topics = []

        for topic_idx, topic in enumerate(self.lda_model.components_):
            top_features_ind = topic.argsort()[: -num_top_words - 1 : -1]
            top_features = [self.feature_names[i] for i in top_features_ind]
            topics.append(top_features)

        return topics

    def transform(self, documents):
        if self.lda_model is None:
            raise Exception("LDA model not fitted.")

        # Convert text into a matrix of token counts
        X = self.vectorizer.transform(documents)

        # Get the topic distribution for each document
        topic_dist = self.lda_model.transform(X)

        return topic_dist

    def display_results(self, topics, topic_dist):
        for idx, topic in enumerate(topics):
            topic_html = f"<h3>Topic #{idx+1}</h3>"
            topic_html += "<ul>"
            for word in topic:
                topic_html += f"<li>{word}</li>"
            topic_html += "</ul>"
            display(HTML(topic_html))

        for i, topic_probs in enumerate(topic_dist):
            sentence_html = f"<h3>Sentence #{i+1}</h3>"
            sentence_html += "<ul>"
            for topic_idx, prob in enumerate(topic_probs):
                sentence_html += f"<li>Topic #{topic_idx+1}: {prob:.4f}</li>"
            sentence_html += "</ul>"
            display(HTML(sentence_html))


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation
# from IPython.display import display, HTML


# class LDA:
#     def __init__(self, n_components=10):
#         self.n_components = n_components
#         self.vectorizer = CountVectorizer()
#         self.lda_model = None
#         self.feature_names = None

#     def fit(self, documents):
#         # Convert text into a matrix of token counts
#         X = self.vectorizer.fit_transform(documents)

#         # Apply Latent Dirichlet Allocation
#         self.lda_model = LatentDirichletAllocation(
#             n_components=self.n_components, random_state=42
#         )
#         self.lda_model.fit(X)

#         # Get the feature names
#         self.feature_names = self.vectorizer.get_feature_names_out()

#     def get_topics(self, num_top_words=5):
#         if self.lda_model is None:
#             raise Exception("LDA model not fitted.")

#         topics = []

#         for topic_idx, topic in enumerate(self.lda_model.components_):
#             top_features_ind = topic.argsort()[: -num_top_words - 1 : -1]
#             top_features = [self.feature_names[i] for i in top_features_ind]
#             topics.append(top_features)

#         return topics

#     def transform(self, documents):
#         if self.lda_model is None:
#             raise Exception("LDA model not fitted.")

#         # Convert text into a matrix of token counts
#         X = self.vectorizer.transform(documents)

#         # Get the topic distribution for each document
#         topic_dist = self.lda_model.transform(X)

#         return topic_dist


# Example usage:


# from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.decomposition import LatentDirichletAllocation


# class LDA:
#     def __init__(self, n_components=10):
#         self.n_components = n_components
#         self.vectorizer = CountVectorizer()
#         self.lda_model = None
#         self.feature_names = None

#     def fit(self, documents):
#         # Convert text into a matrix of token counts
#         X = self.vectorizer.fit_transform(documents)

#         # Apply Latent Dirichlet Allocation
#         self.lda_model = LatentDirichletAllocation(
#             n_components=self.n_components, random_state=42
#         )
#         self.lda_model.fit(X)

#         # Get the feature names
#         self.feature_names = self.vectorizer.get_feature_names_out()

#     def get_topics(self, num_top_words=5):
#         if self.lda_model is None:
#             raise Exception("LDA model not fitted.")

#         topics = []

#         for topic_idx, topic in enumerate(self.lda_model.components_):
#             top_features_ind = topic.argsort()[: -num_top_words - 1 : -1]
#             top_features = [self.feature_names[i] for i in top_features_ind]
#             topics.append(top_features)

#         return topics

#     def transform(self, documents):
#         if self.lda_model is None:
#             raise Exception("LDA model not fitted.")

#         # Convert text into a matrix of token counts
#         X = self.vectorizer.transform(documents)

#         # Get the topic distribution for each document
#         topic_dist = self.lda_model.transform(X)

#         return topic_dist


# # Example usage:
# documents = [
#     "I love playing soccer",
#     "I enjoy watching movies",
#     "Football is my favorite sport",
#     "I like going to the gym",
#     "I prefer reading books over watching TV",
#     "I am a fan of basketball",
# ]

# lda = LDA(n_components=2)
# lda.fit(documents)

# topics = lda.get_topics(num_top_words=3)
# for idx, topic in enumerate(topics):
#     print(f"Topic #{idx+1}: {', '.join(topic)}")

# topic_dist = lda.transform(documents)
# for i, topic_probs in enumerate(topic_dist):
#     print(f"Sentence #{i+1}:")
#     for topic_idx, prob in enumerate(topic_probs):
#         print(f"Topic #{topic_idx+1}: {prob:.4f}")
#     print()
