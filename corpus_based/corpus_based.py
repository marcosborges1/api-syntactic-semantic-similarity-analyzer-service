from abc import abstractmethod
from IPython.display import display, HTML
import pandas as pd


class CorpusBased:
    # def __init__(self, name):
    #     self.name = name

    # @abstractmethod
    # def analyze(self):
    #     pass

    # def get_dataframe(self, results):
    #     columns = ["word1", "word2"] + self.similarity_measures
    #     df = pd.DataFrame(results, columns=columns)
    #     return df

    # def to_string(self, results):
    #     df = self.get_dataframe(results)
    #     print(df.to_string(index=False))

    # def display_html(self, results):
    #     df = self.get_dataframe(results)
    #     display(HTML(df.to_html(index=False)))

    def __init__(self, name):
        self.name = name
        self.columns = ["word1", "word2"] # Header based on word_list content
        self.results = []  # Results of Analyze method
        self.similarity_measures = [] 
        self.df = []

    def analyze(self, word_list):
        raise NotImplementedError("Subclasses must implement analyze() method")
    
    def set_columns(self, columns):
        self.columns = columns

    def get_results(self):
        return self.results
    
    def generate_dataframe(self):
        self.df = pd.DataFrame(self.results, columns=self.columns + self.similarity_measures)
    
    def get_dataframe(self):
        return self.df
    
    def display_string(self):
        print(self.get_dataframe().to_string(index=False))

    def display_html(self):
        display(HTML(self.get_dataframe().to_html(index=False)))