#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from Bio import Entrez
from transformers import BertTokenizer, BertModel, AutoTokenizer
import torch
from tqdm import tqdm
import time
import matplotlib.pyplot as plt


# In[2]:


# Importation des PMID

pmid_csv = pd.read_csv('corpus_pubmed_ids_for_missing_abstracts_filtered_filtered_prepared.csv')

pmid_list = pmid_csv['pubmedid']  # La liste des PMID

print(pmid_list.head())


# In[3]:


# Contrôle des PMID
duplicate_rows = pmid_list[pmid_list.duplicated()]

print("Les lignes en doublon :")
print(duplicate_rows)


# In[7]:


pmid_list = pmid_list.to_numpy()  # Convertie en np array

pmid_list = np.transpose(pmid_list)

print('la liste est de la taille : ', len(pmid_list), 'et de shape : ', pmid_list.shape)

print(pmid_list)


# In[9]:


# Plusieurs règles à suivre : Fournir mon email, éviter un grand nombre de query 9-17h américain
# en jours de semaine et pas plus de 3 query per second

# Email

Entrez.email = "mathissommacal2@gmail.com"

# Obtenir le nom des bases de données accessible via Entrez

handle = Entrez.einfo()  # ouvrir une connection au serveur

rec = Entrez.read(handle)  # Lire la requête

handle.close()  # Fermer la connection

print(rec.keys())

data_base = rec['DbList']

print(data_base)

print('On a la base de donnée qui nous interesse : ', data_base[0])


# In[91]:


# Récupérer les données, infos des artictles cibles


def get_pubmed_data(pmids):
    articles_data = []
    pmid_ok = []
    pmid_error = []

    for pmid in tqdm(pmids, desc="Récupération des articles"):
        try:
            handle = Entrez.efetch(db="pubmed", id=pmid, rettype="abstract", retmode="xml") # extrait en fichier xml
            records = Entrez.read(handle)

            # Récupération des infos de l'article
            article = records['PubmedArticle'][0] # récupérer le premier article correspondant à la recherche
            article_title = article['MedlineCitation']['Article']['ArticleTitle'] # accéder au titre
            abstract = article['MedlineCitation']['Article']['Abstract']['AbstractText'][0] # accéder à l'extrait
            authors = article['MedlineCitation']['Article']['AuthorList'] # accéder aux auteurs

            # récupération des nom des auteurs
            author_names = []
            for author in authors:
                try:
                    author_names.append(f"{author['ForeName']} {author['LastName']}")
                except
                    pass  # Ignorer si pas d'auteurs

            # Ajouter les informations de l'article aux listes
            pmid_ok.append(pmid) # Pour récupérer les pmid des articles avec extraits
            articles_data.append({
                'pmid': pmid,
                'title': article_title,
                'abstract': abstract,
                'authors': ", ".join(author_names)  # join pour fusionner le noms des auteurs pour chaque article
            })

        except Exception as e:
            pmid_error.append(pmid) # récupérer les pmid des articles qui n'ont pas d'extrait
            articles_data.append({
                'pmid': pmid,
                'title': article_title,
                'abstract': '',
                'authors': ", ".join(author_names)  # Fusionner les noms des auteurs
            })
            continue
    
    # Création du DataFrame
    articles_df = pd.DataFrame(articles_data)
    
    return articles_df, pmid_ok, pmid_error

# Utilisation avec un nombre correct de pmid (qui respect les règles)
pmid_list = pmid_list.astype(str) # Changement en type caractères 'str'
tr_pmid = pmid_list[0:200] # Le nombre de pmid cibles à récupérer pour la requête
print('Liste des PMIDs :', tr_pmid)
pmids = tr_pmid

# Appel de la fonction avec retour des df/listes
articles_df, pmid_ok, pmid_error = get_pubmed_data(pmids)


# In[57]:


articles_df.head()


# In[121]:


# Analyse des données récupérées (si possède infos : nom d'auteur/extrait/titre ..)
def analyze_data_quality(articles_df):
    articles_df['has_abstract'] = articles_df['abstract'].apply(lambda x: 1 if (len(x)>2) else 0) # booleen 1 si il y a un extrait (je considère qu'il y a un extrait si plus de 2 caractères)
    articles_df['abstract_length'] = articles_df['abstract'].apply(len) # Pour avoir le nombre de caractère des extraits si existe
    articles_df['has_authors'] = articles_df['authors'].apply(lambda x: 1 if x else 0) # Savoir si l'article mentionne les auteurs
    articles_df['has_title'] = articles_df['title'].apply(lambda x: 1 if x else 0) # savoir si l'article à un titre

    
    # Print les resultats avec statistiques
    print("Résumé des données :")
    print(articles_df.describe())
    
    # Histo pour visualiser le nombre d'extrait en fonction de la longueur de l'extrait
    plt.hist(articles_df['abstract_length'], bins=50, color='blue', alpha=0.7)
    plt.title("Distribution des longueurs de résumés")
    plt.xlabel("Longueur du résumé")
    plt.ylabel("Nombre d'articles")
    plt.show()


# In[125]:


pmid_oka = np.array(pmid_ok)
pmid_oka = pmid_oka.astype(int)
pmid_errora = np.array(pmid_error)
pmid_errora = pmid_errora.astype(int)

print('Il y a : ',len(pmid_ok),' extraits d articles importés')
print('Il y a : ',len(pmid_error),' extraits d articles non importés')

# Camembert pour visualiser le % d'article avec et sans extrait
plt.figure(figsize = (8, 8))
x = [len(pmid_ok),len(pmid_error)] # array contenant les tailles à visualiser sur le camembert

plt.pie(x, labels = ['extraits d articles importés', 'extraits d articles non importés'],
           colors = ['green', 'red'],
           explode = [0, 0.2], # Pour décaler la 2 ieme tranche
           autopct = lambda x: str(round(x, 2)) + '%', # affichage des pourcentages
           pctdistance = 0.5, labeldistance = 1.2, # poitionnement des % et des labels
           shadow = True) 
plt.legend()


# In[127]:


# Analyse qualité des données
analyze_data_quality(articles_df) # Appel de la fonction d'analyse

print('On observe que chaque article à un titre et des auteurs mais seulement ',len(pmid_ok),'/',len(pmids),' ont un extrait')


# In[65]:


# Quels auteurs ont le plus d'articles sur les pmid cibles?

# Split des auteurs
articles_df['authors_list'] = articles_df['authors'].str.split(', ') # On split à chaque (,space)

# On fait une liste à partir des auteurs de chaque article pour avoir un auteur par ligne
all_authors = articles_df.explode('authors_list') # Si un article contient plusieurs auteurs creer une nouvelle ligne pour chaque auteur

# Compter le nombre d'articles par auteur
author_article_counts = all_authors['authors_list'].value_counts()

top_authors = author_article_counts.head(10) # top 10 avec le plus d'article

print("Top 10 des auteurs avec le plus grand nombre d'articles :")
print(top_authors)

# Histogramme du top 10
plt.figure(figsize=(10, 6))
top_authors.plot(kind='bar', color='red')
plt.title("Nombre d'articles par auteur (Top 10)")
plt.xlabel("Auteurs")
plt.ylabel("Nombre d'articles")
plt.show()


# In[ ]:


# 1 ière méthode de vectorisation


# In[67]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import PCA


# In[69]:


def vectorize_data(articles_df):
    tfidf = TfidfVectorizer(stop_words='english', max_features=5000) # mots très courants comme ("the", "is", "and") seront ignorés puis limite de 5000 mots pour éviter trop grand vecteurs
    vectors = tfidf.fit_transform(articles_df['abstract']) # vectorisation TF-IDF (Term Frequency - Inverse Document Frequency)
    
    # Réduction de dimension avec PCA (Principal Composent Analysis)
    pca = PCA(n_components=2) # On passe de tenseur en dimension 5000 ou - à 2D pour visualiser les vecteurs sur graphique 2D
    reduced_vectors = pca.fit_transform(vectors.toarray())
    
    # Visualisation 2D
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1], alpha=0.5)
    plt.title("Projection 2D des articles vectorisés (PCA)")
    plt.show()
    return vectors


# In[71]:


# Vectorisation des données
vectors = vectorize_data(articles_df)


# In[129]:


# On observe un cluster ce qui signifie que les extrait des articles parlent surement de sujet similaire (context similaire)
print(vectors.shape)


# In[75]:


# 2 ième méthode avec transformers de huggingface afin d'utiliser le model BERT
extrait = articles_df['abstract'].to_list()
print('Il y a : ',len(extrait),'extraits importés')
print(type(extrait))


# In[77]:


# Tokenize l'extrait avec transformers de hugging face

tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased", clean_up_tokenization_spaces = True)

model = BertModel.from_pretrained('bert-base-uncased', output_attentions =True, attn_implementation="eager")

# Tokenisation du texte (padding avec les tokens spéciaux [CLS], [SEP] et tronque si trop petit pour avoir les mêmes tailles de vecteurs)
inputs = tokenizer(extrait, return_tensors='pt', padding=True, truncation=True, max_length=128)

# Générer les embeddings en passant les tokens dans le modèle BERT
with torch.no_grad():  # sans le calcul des gradients
    outputs = model(**inputs)

# Les embeddings sont dans outputs.last_hidden_state
# Cela retourne un tenseur de forme (batch_size, seq_length, hidden_size)
embeddings = outputs.last_hidden_state

# Extraire l'embedding de la token [CLS] pour chaque phrase (premier token)
cls_embeddings = embeddings[:, 0, :]  # (batch_size, hidden_size)

print(cls_embeddings.shape)  # (nombre_de_textes, taille_des_embeddings)
print(cls_embeddings)


# In[83]:


import seaborn as sns
from sklearn.manifold import TSNE
import plotly.graph_objs as go


# In[79]:


# outputs contient les résultats de BERT avec l'attention.
attentions = outputs.attentions[-1]  # Attention de la dernière couche

# Moyenne des attentions sur toutes les têtes pour simplifier l'analyse
mean_attention = attentions.mean(dim=1).squeeze()

# Vérification de la dimension des 'input_ids' pour s'assurer qu'on travaille avec une seule séquence
input_ids = inputs['input_ids'].squeeze()

# Si le batch contient plusieurs séquences, on traite chaque séquence individuellement
if input_ids.dim() > 1:
    # Sélectionner la première séquence pour l'exemple
    input_ids = input_ids[0]

# Convertir les 'input_ids' en tokens pour la visualisation
tokens = tokenizer.convert_ids_to_tokens(input_ids.tolist())

# Afficher les tokens
print(tokens)

# Afficher les scores d'attention moyens
print(mean_attention)


# In[85]:


# Moyenne des attentions sur toutes les positions de chaque mot (en supposant une dimension [batch_size, sequence_length, sequence_length])
# Cela réduit la matrice à une taille de [batch_size, sequence_length]
mean_attention_2d = mean_attention.mean(dim=-1)

# Réduction de dimension avec t-SNE pour obtenir des données 3D
# Choisir une perplexité adaptée à la taille des données (doit être < nombre d'échantillons)
perplexity_value = min(5, mean_attention_2d.shape[0] - 1)  # Adaptation dynamique
tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity_value)
reduced_attention = tsne.fit_transform(mean_attention_2d.cpu().numpy())

# Vérifie si 'input_ids' contient plusieurs séquences ou une seule
input_ids = inputs['input_ids'].squeeze().tolist()  # Supprime les dimensions inutiles

# Si input_ids est une liste de listes (plusieurs séquences), on doit itérer sur chaque séquence
if isinstance(input_ids[0], list):
    tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in input_ids]  # Conversion pour chaque séquence
else:
    tokens = tokenizer.convert_ids_to_tokens(input_ids)  # Conversion pour une seule séquence

# Préparer la visualisation avec Plotly
trace = go.Scatter3d(
    x=reduced_attention[:, 0],
    y=reduced_attention[:, 1],
    z=reduced_attention[:, 2],
    mode='markers+text',
    marker=dict(size=3, color=np.arange(len(tokens)), colorscale='thermal', opacity=0.8),
    text=tokens,  # Annoter les points avec les tokens
    textposition='top center'
)

# Configuration du layout du graphique
layout = go.Layout(
    title="Visualisation 3D des mots basée sur l'attention",
    scene=dict(
        xaxis_title="Dimension 1",
        yaxis_title="Dimension 2",
        zaxis_title="Dimension 3"
    )
)

# Créer la figure et l'afficher
fig = go.Figure(data=[trace], layout=layout)
fig.show()


# In[87]:


import hnswlib


# In[89]:


data = cls_embeddings  
# Initialiser l'index HNSW (Hierachical Navigable Small Worlds)
dim = 768  # Dimension des vecteurs
num_elements = 200  # Nombre de vecteurs ici nombre d'extraits

hnsw_index = hnswlib.Index(space='l2', dim=dim)  # distance euclidienne (L2) pour mesurer la similarité entre les vecteurs.

# Initialisation de l'index
hnsw_index.init_index(max_elements=num_elements, ef_construction=200, M=16) # max_elements : Le nombre maximum d'éléments que l'index peut contenir.
# ef_construction : paramètre qui influence la qualité de l'index lors de sa construction plus il est grand  meilleure est la qualité, mais plus de temps.
# M : Contrôle la taille des connexions entre les noeuds dans l'index, valeur plus élevée conduit à meilleur rappel mais recherche plus long.

# Ajouter les vecteurs
hnsw_index.add_items(data)

# Définir le paramètre de recherche (nombre de voisins à examiner)
hnsw_index.set_ef(50)  # ef doit être grand pour plus de précision

# Vecteur de la requête afin de trouver la réponse dans le datastore
query_vector = data[0] # ici juste le premier vecteur embedded
labels, distances = hnsw_index.knn_query(query_vector, k=10) # Obtenir les résultats en consultant la datastore

print("Labels des voisins les plus proches:", labels)
print("Distances correspondantes:", distances)

