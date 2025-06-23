import pandas as pd
from IPython.display import display, HTML


class DeepLearningBased:
    def __init__(self, name, dataset):
        """
        Classe base para Deep Learning.
        
        Parâmetros:
        - name (str): O nome da classe.
        """
        self.name = name
        self.dataset = None
        self.similarity_measures = []  # Agora será usado para armazenar models_to_compute dinamicamente

        # Carregar o arquivo data.csv automaticamente
        try:
            # self.dataset = pd.read_csv('data.csv')
            self.dataset = dataset
            # print("Arquivo 'data.csv' carregado com sucesso!")
        except FileNotFoundError:
            raise FileNotFoundError("O arquivo 'data.csv' não foi encontrado no diretório. Verifique o caminho e tente novamente.")
        except Exception as e:
            raise Exception(f"Erro ao carregar o arquivo 'data.csv': {e}")

    def display_string(self):
        """Exibe o DataFrame de forma tabular no terminal."""
        if self.dataset is not None:
            print(self.dataset.to_string(index=False))
        else:
            print("Dataset não carregado.")

    def display_html(self):
        """Exibe o DataFrame como HTML (para Jupyter Notebooks)."""
        if self.dataset is not None:
            display(HTML(self.dataset.to_html(index=False)))
        else:
            print("Dataset não carregado.")