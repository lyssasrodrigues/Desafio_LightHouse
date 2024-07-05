# Análise de Dados Cinematográficos

Este projeto realiza uma análise exploratória de dados (EDA), visualização e modelagem preditiva em um conjunto de dados de filmes. Ele fornece uma interface gráfica para explorar e analisar os dados de maneira interativa.

## Estrutura do Projeto

- `desafio_indicium_imdb.csv`: Arquivo CSV contendo os dados dos filmes.
- `script.py`: Script Python principal que realiza a análise e apresenta uma interface gráfica interativa.
- `requirements.txt`: Lista de dependências do projeto.

## Instalação

1. Clone o repositório:
    ```bash
    git clone https://github.com/lyssasrodrigues/Desafio_LightHouse.git
    cd seu-repositorio
    ```

2. Crie um ambiente virtual e ative-o:
    ```bash
    python -m venv venv
    source venv/bin/activate  # No Windows, use `venv\Scripts\activate`
    ```

3. Instale as dependências:
    ```bash
    pip install -r requirements.txt
    ```

## Execução

Para executar o script Python e abrir a interface gráfica, use o seguinte comando:
```bash
python script.py desafio_indicium_imdb.csv
```

## Funcionalidades

- **Exploração de Dados (EDA)**: A análise exploratória de dados (EDA) é um passo fundamental em qualquer projeto de ciência de dados. O script realiza EDA apresentando visualizações como distribuições de anos de lançamento e notas do IMDB.
  
- **Distribuição de Gêneros**: Mostra a distribuição dos gêneros dos filmes, fornecendo insights sobre quais são os mais comuns no dataset.
  
- **Análise de Sinopses**: Gera uma nuvem de palavras das sinopses dos filmes e agrupa os filmes em clusters com base em suas sinopses.
  
- **Recomendação de Filme**: Recomenda o filme com a maior nota do IMDB presente no dataset.
  
- **Fatores de Alto Faturamento**: Identifica as características que mais influenciam o faturamento dos filmes, usando um modelo de regressão linear.
  
- **Previsão da Nota do IMDB**: Utiliza um modelo de RandomForestRegressor para prever a nota do IMDB de um filme com base em suas características.

## Respostas às Perguntas

### Análise Exploratória dos Dados (EDA)

A EDA revela várias características importantes dos dados. Por exemplo:
- **Distribuição dos Anos de Lançamento**: Mostra que a maioria dos filmes no dataset foram lançados nos últimos 30 anos.
- **Distribuição das Notas do IMDB**: Revela que a maioria dos filmes tem notas entre 6 e 8 no IMDB.

### Qual filme você recomendaria para uma pessoa que você não conhece?

Recomendaríamos o filme com a maior nota no IMDB do dataset. No caso deste dataset, seria o filme *The Shawshank Redemption*.

### Quais são os principais fatores que estão relacionados com alta expectativa de faturamento de um filme?

Os principais fatores relacionados ao alto faturamento são:
- **Meta_score**: Filmes com maiores metascores tendem a ter maior faturamento.
- **Número de Votos no IMDB**: Filmes com mais votos tendem a ser mais populares e, portanto, têm maior faturamento.

### Quais insights podem ser tirados com a coluna Overview? É possível inferir o gênero do filme a partir dessa coluna?

A coluna Overview pode fornecer insights significativos através da análise de texto. Usando técnicas de processamento de linguagem natural (NLP), como TF-IDF e clustering, é possível agrupar filmes em gêneros semelhantes. 
Embora não seja perfeitamente preciso, é possível inferir o gênero do filme a partir das sinopses com uma taxa de acerto razoável.

### Explicação da Previsão da Nota do IMDB

Para prever a nota do IMDB:
- **Variáveis Utilizadas**: Meta_score, No_of_Votes, Gross
- **Transformações**: Normalização das variáveis para melhorar o desempenho do modelo.
- **Tipo de Problema**: Regressão, pois estamos prevendo um valor contínuo (nota do IMDB).
- **Modelo Utilizado**: RandomForestRegressor. Este modelo é escolhido por sua robustez e capacidade de lidar com variáveis não-lineares.
- **Medida de Performance**: Erro Quadrático Médio (MSE), pois mede a média dos quadrados dos erros, penalizando predições distantes dos valores reais.

### Previsão da Nota para um Filme Específico

Para prever a nota do filme *The Shawshank Redemption* com as características fornecidas, o modelo treinado seria utilizado da seguinte forma:
```python
filme = {
    'Series_Title': 'The Shawshank Redemption',
    'Released_Year': '1994',
    'Certificate': 'A',
    'Runtime': '142 min',
    'Genre': 'Drama',
    'Overview': 'Two imprisoned men bond over a number of years, finding solace and eventual redemption through acts of common decency.',
    'Meta_score': 80.0,
    'Director': 'Frank Darabont',
    'Star1': 'Tim Robbins',
    'Star2': 'Morgan Freeman',
    'Star3': 'Bob Gunton',
    'Star4': 'William Sadler',
    'No_of_Votes': 2343110,
    'Gross': '28,341,469'
}

prever_nota_filme(filme)
```

## Salvando o Modelo

O modelo RandomForestRegressor treinado é salvo no formato `.pkl` para uso futuro.
```python
with open('modelo_nota_imdb.pkl', 'wb') as arquivo:
    pickle.dump(modelo, arquivo)
```

## Contribuições

Contribuições são bem-vindas! Se você deseja contribuir com este projeto, por favor, siga estas etapas:
1. Faça um fork do repositório.
2. Crie uma nova branch (`git checkout -b feature/MinhaNovaFeature`).
3. Faça commit das suas alterações (`git commit -m 'Adiciona uma nova feature'`).
4. Faça push para a branch (`git push origin feature/MinhaNovaFeature`).
5. Abra um Pull Request.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## Contato

Se você tiver alguma dúvida ou sugestão, entre em contato:
- Nome: [Lyssa Rodrigues]
- Email: [lyssa.rodrigues@upe.br]
