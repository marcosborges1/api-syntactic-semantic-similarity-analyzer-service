import warnings
import torch

# Suprimir o aviso "TypedStorage is deprecated"
warnings.filterwarnings("ignore", message="TypedStorage is deprecated")

from .deep_learning import DeepLearningBased
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM  # Usando o modelo correto
import numpy as np

class MyTransformers(DeepLearningBased):
    def __init__(self, name, dataset):
        """
        Classe que herda de DeepLearning e implementa os métodos para análise de similaridade.
        
        Parâmetros:
        - name (str): O nome da classe.
        """
        super().__init__(name,dataset)
        self.models = {}  # Dicionário para armazenar os modelos sob demanda

    def load_model(self, model_name):
        """
        Carrega o modelo de forma sob demanda.
        
        Parâmetros:
        - model_name (str): Nome do modelo a ser carregado ('bert', 'roberta', 't5' ou 'bart').
        
        Retorno:
        - pipeline: O modelo carregado.
        """
        if model_name in self.models:
            print(f"Modelo '{model_name}' já está carregado.")
            return self.models[model_name]
        
        print(f"Carregando o modelo '{model_name}' sob demanda...")
        if model_name == "bert":
            self.models[model_name] = pipeline("feature-extraction", model="bert-base-uncased")
        elif model_name == "roberta":
            self.models[model_name] = pipeline("feature-extraction", model="sentence-transformers/all-roberta-large-v1")
        elif model_name == "t5":
            tokenizer = AutoTokenizer.from_pretrained("t5-base", model_max_length=512)
            model = AutoModelForSeq2SeqLM.from_pretrained("t5-base")
            self.models[model_name] = {'tokenizer': tokenizer, 'model': model}
        elif model_name == "bart":
            self.models[model_name] = pipeline("feature-extraction", model="facebook/bart-large")
        else:
            raise ValueError(f"Modelo '{model_name}' não está disponível. Modelos disponíveis: ['bert', 'roberta', 't5', 'bart']")
        
        return self.models[model_name]

    def calculate_similarity(self, model, desc1, desc2, model_name):
        """
        Calcula a similaridade de cosseno entre duas descrições.
        
        Parâmetros:
        - model (pipeline ou dict): O modelo de embeddings (BERT, RoBERTa, T5 ou BART).
        - desc1 (str): Descrição do campo 1.
        - desc2 (str): Descrição do campo 2.
        - model_name (str): Nome do modelo.
        
        Retorno:
        - float: Similaridade do cosseno entre os dois embeddings.
        """
        try:
            if model_name == "t5":
                tokenizer = model['tokenizer']
                model = model['model']
                
                # Codificar as frases com truncamento e padding
                inputs1 = tokenizer(desc1, return_tensors="pt", truncation=True, padding=True, max_length=512)
                inputs2 = tokenizer(desc2, return_tensors="pt", truncation=True, padding=True, max_length=512)
                
                with torch.no_grad():
                    output1 = model.encoder(inputs1.input_ids).last_hidden_state
                    output2 = model.encoder(inputs2.input_ids).last_hidden_state
                
                output1 = torch.mean(output1, dim=1)
                output2 = torch.mean(output2, dim=1)
                
                if torch.all(output1 == 0) or torch.all(output2 == 0):
                    print(f"⚠️ Embeddings nulos para as descrições: '{desc1}' e '{desc2}'")
                    return 0.0
                
                output1 = output1 / torch.norm(output1, p=2, dim=1, keepdim=True)
                output2 = output2 / torch.norm(output2, p=2, dim=1, keepdim=True)
                
                output1 = output1.cpu().detach().numpy()
                output2 = output2.cpu().detach().numpy()
                
                similarity = cosine_similarity(output1, output2)[0][0]
            else:
                desc1_embedding = np.mean(model(desc1), axis=1)
                desc2_embedding = np.mean(model(desc2), axis=1)
                similarity = cosine_similarity([desc1_embedding[0]], [desc2_embedding[0]])[0][0]
            
            return round(similarity, 3)
        except Exception as e:
            print(f"Erro ao calcular a similaridade: {e}")
            return 0.0

    def analyze(self, comparison_fields, models_to_compute):
        """
        Executa a análise de similaridade de acordo com os modelos e campos especificados.
        
        Parâmetros:
        - comparison_fields (list): Lista de campos que serão comparados.
        - models_to_compute (list): Lista de nomes de modelos para cálculo de similaridade.
        
        Retorno:
        - pd.DataFrame: DataFrame atualizado com as colunas de similaridade.
        """
        if not models_to_compute:
            raise ValueError("O array de modelos está vazio. Informe ao menos um modelo.")
        
        if len(comparison_fields) != 2:
            raise ValueError("comparison_fields deve conter exatamente 2 campos.")
        
        self.similarity_measures = models_to_compute
        field1, field2 = comparison_fields

        for model_name in models_to_compute:
            if model_name in ['bert', 'roberta', 't5', 'bart']:
                model = self.load_model(model_name)
                
                desc1_list = self.dataset[field1].tolist()
                desc2_list = self.dataset[field2].tolist()
                
                print(f"Calculando similaridades usando o modelo '{model_name}' para os campos '{field1}' e '{field2}'...")
                similarities = [
                    self.calculate_similarity(model, desc1, desc2, model_name) 
                    for desc1, desc2 in zip(desc1_list, desc2_list)
                ]
                
                self.dataset[model_name] = similarities
            else:
                print(f"Modelo '{model_name}' não está disponível. Ignorando...")

        return self.dataset