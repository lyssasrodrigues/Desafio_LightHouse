import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import pickle
import sys
import tkinter as tk
from tkinter import ttk

def carregar_e_limpar_dados(caminho_arquivo):
    try:
        df = pd.read_csv(caminho_arquivo)
        df.dropna(inplace=True)
        df = df[pd.to_numeric(df['Released_Year'], errors='coerce').notnull()]
        df['Released_Year'] = df['Released_Year'].astype(int)
        df['Gross'] = df['Gross'].str.replace(',', '').astype(float)
        df['Meta_score'] = df['Meta_score'].astype(float)
        return df
    except FileNotFoundError:
        print("Erro: Arquivo não encontrado.")
        sys.exit(1)
    except Exception as e:
        print(f"Erro ao processar o arquivo: {e}")
        sys.exit(1)

def mostrar_dataframe(df, titulo):
    root = tk.Toplevel()
    root.title(titulo)
    frame = ttk.Frame(root)
    frame.pack(fill=tk.BOTH, expand=True)
    
    tree = ttk.Treeview(frame, columns=list(df.columns), show='headings')
    for col in df.columns:
        tree.heading(col, text=col)
        tree.column(col, anchor=tk.CENTER)
    for row in df.itertuples(index=False):
        tree.insert("", tk.END, values=row)
    
    tree.pack(fill=tk.BOTH, expand=True)
    root.mainloop()

def eda(df):
    mostrar_dataframe(df.head(), "Primeiros 5 Registros")
    mostrar_dataframe(df.describe().transpose(), "Resumo Estatístico")
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['Released_Year'], kde=True)
    plt.title('Distribuição dos Anos de Lançamento')
    plt.xlabel('Ano de Lançamento')
    plt.ylabel('Frequência')
    plt.show()
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df['IMDB_Rating'], bins=20, kde=True)
    plt.title('Distribuição das Notas do IMDB')
    plt.xlabel('Nota do IMDB')
    plt.ylabel('Frequência')
    plt.show()

def plotar_distribuicao_generos(df):
    todos_generos = df['Genre'].str.split(',').explode().str.strip()
    contagem_generos = todos_generos.value_counts()
    plt.figure(figsize=(10, 10))
    sns.barplot(y=contagem_generos.index, x=contagem_generos.values, orient='h')
    plt.title('Distribuição de Gêneros')
    plt.ylabel('Gênero')
    plt.xlabel('Contagem')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def plotar_distribuicao_notas_imdb(df):
    plt.figure(figsize=(12, 6))
    sns.histplot(df['IMDB_Rating'], bins=20, kde=True)
    plt.title('Distribuição das Notas do IMDB')
    plt.xlabel('Nota do IMDB')
    plt.ylabel('Frequência')
    plt.show()

def plotar_meta_score_vs_faturamento(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='Meta_score', y='Gross', data=df)
    plt.title('Correlação entre Meta_score e Faturamento')
    plt.xlabel('Meta_score')
    plt.ylabel('Faturamento')
    plt.show()

def plotar_numero_votos_vs_nota_imdb(df):
    plt.figure(figsize=(12, 6))
    sns.scatterplot(x='No_of_Votes', y='IMDB_Rating', data=df)
    plt.title('Correlação entre Número de Votos e Nota do IMDB')
    plt.xlabel('Número de Votos')
    plt.ylabel('Nota do IMDB')
    plt.show()

def analisar_sinopses(df):
    texto = " ".join(review for review in df['Overview'])
    nuvem_palavras = WordCloud(max_words=100, background_color="white").generate(texto)
    plt.figure(figsize=(10, 5))
    plt.imshow(nuvem_palavras, interpolation='bilinear')
    plt.axis("off")
    plt.title('Nuvem de Palavras das Sinopses')
    plt.show()

    vetorizar = TfidfVectorizer(stop_words='english')
    X = vetorizar.fit_transform(df['Overview'])

    kmeans = KMeans(n_clusters=5, random_state=42)
    kmeans.fit(X)
    df['Cluster'] = kmeans.labels_

    visao_geral_clusters = df.groupby('Cluster')['Genre'].apply(lambda x: x.mode()[0])
    mostrar_dataframe(visao_geral_clusters.reset_index(), "Visão Geral dos Clusters")

def recomendar_filme(df):
    filme_recomendado = df[df['IMDB_Rating'] == df['IMDB_Rating'].max()]['Series_Title'].iloc[0]
    root = tk.Toplevel()
    root.title("Filme Recomendado")
    label = ttk.Label(root, text=f"Filme recomendado: {filme_recomendado}", font=("Helvetica", 16))
    label.pack(pady=20)
    root.mainloop()

def fatores_alto_faturamento(df):
    X = df[['Meta_score', 'No_of_Votes', 'IMDB_Rating']]
    y = df['Gross']

    modelo = LinearRegression()
    modelo.fit(X, y)

    importancias = modelo.coef_
    caracteristicas = X.columns
    importancia_caracteristicas = pd.DataFrame({'Característica': caracteristicas, 'Importância': importancias})
    importancia_caracteristicas = importancia_caracteristicas.sort_values(by='Importância', ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importância', y='Característica', data=importancia_caracteristicas)
    plt.title('Importância das Características para o Faturamento')
    plt.xlabel('Importância')
    plt.ylabel('Característica')
    plt.show()

    mostrar_dataframe(importancia_caracteristicas, "Importância das Características")

def insights_sinopses(df):
    vetorizar = TfidfVectorizer(stop_words='english')
    X = vetorizar.fit_transform(df['Overview'])
    y = df['Genre']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = MultinomialNB()
    modelo.fit(X_train, y_train)

    y_pred = modelo.predict(X_test)
    acuracia = accuracy_score(y_test, y_pred)
    relatorio = classification_report(y_test, y_pred, output_dict=True)
    relatorio_df = pd.DataFrame(relatorio).transpose()
    
    root = tk.Toplevel()
    root.title("Classificação de Gênero")
    label = ttk.Label(root, text=f"Acurácia: {acuracia}", font=("Helvetica", 16))
    label.pack(pady=20)
    mostrar_dataframe(relatorio_df, "Relatório de Classificação")

def prever_nota_imdb(df):
    caracteristicas = ['Meta_score', 'No_of_Votes', 'Gross']
    X = df[caracteristicas]
    y = df['IMDB_Rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    modelo = RandomForestRegressor(n_estimators=100, random_state=42)
    modelo.fit(X_train, y_train)
    y_pred = modelo.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    
    root = tk.Toplevel()
    root.title("Erro Quadrático Médio")
    label = ttk.Label(root, text=f'Erro Quadrático Médio: {mse}', font=("Helvetica", 16))
    label.pack(pady=20)
    
    plt.figure(figsize=(12, 6))
    plt.scatter(y_test, y_pred)
    plt.xlabel("Valores Reais")
    plt.ylabel("Predições")
    plt.title("Valores Reais vs Predições")
    plt.show()

    root.mainloop()

    with open('modelo_nota_imdb.pkl', 'wb') as arquivo:
        pickle.dump(modelo, arquivo)

def prever_nota_filme(filme, modelo_path='modelo_nota_imdb.pkl'):
    with open(modelo_path, 'rb') as arquivo:
        modelo = pickle.load(arquivo)
    
    dados_filme = np.array([[filme['Meta_score'], filme['No_of_Votes'], filme['Gross']]])
    nota_predita = modelo.predict(dados_filme)
    print(f'A nota prevista para o filme {filme["Series_Title"]} é: {nota_predita[0]}')

def principal(caminho_arquivo):
    df = carregar_e_limpar_dados(caminho_arquivo)
    
    root = tk.Tk()
    root.title("Análise de Dados Cinematográficos")
    root.geometry("400x500")
    
    titulo = ttk.Label(root, text="Análise de Dados Cinematográficos", font=("Helvetica", 18))
    titulo.pack(pady=20)
    
    def ao_eda():
        eda(df)
    
    def ao_analisar_sinopses():
        analisar_sinopses(df)
    
    def ao_recomendar_filme():
        recomendar_filme(df)
    
    def ao_fatores_alto_faturamento():
        fatores_alto_faturamento(df)
    
    def ao_insights_sinopses():
        insights_sinopses(df)
    
    def ao_prever_nota_imdb():
        prever_nota_imdb(df)
    
    botoes = [
        ("Exploração de Dados (EDA)", ao_eda),
        ("Distribuição de Gêneros", lambda: plotar_distribuicao_generos(df)),
        ("Distribuição de Notas IMDB", lambda: plotar_distribuicao_notas_imdb(df)),
        ("Meta_score vs Faturamento", lambda: plotar_meta_score_vs_faturamento(df)),
        ("Número de Votos vs Nota IMDB", lambda: plotar_numero_votos_vs_nota_imdb(df)),
        ("Analisar Sinopses", ao_analisar_sinopses),
        ("Recomendar Filme", ao_recomendar_filme),
        ("Fatores de Alto Faturamento", ao_fatores_alto_faturamento),
        ("Insights das Sinopses", ao_insights_sinopses),
        ("Prever Nota IMDB", ao_prever_nota_imdb)
    ]
    
    for texto, comando in botoes:
        botao = ttk.Button(root, text=texto, command=comando)
        botao.pack(pady=5)
    
    root.mainloop()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        caminho_arquivo = sys.argv[1]
        principal(caminho_arquivo)
    else:
        print("Uso: python script.py <caminho_para_arquivo_csv>")
