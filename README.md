# Perceptron do Zero com Python e NumPy

Implementação didática de um **Perceptron binário**, um dos modelos fundamentais do aprendizado de máquina supervisionado. O projeto apresenta treinamento, previsões, visualização das fronteiras de decisão e testes automatizados, utilizando Python e NumPy para implementar o algoritmo sem depender de bibliotecas de modelos prontos.

> **Transparência:** o repositório original enviado para revisão continha somente um notebook com importações de NumPy e Matplotlib e um README com o título. O algoritmo, as demonstrações e os testes deste repositório foram criados durante a reorganização. Confira [Origem e alterações](docs/ORIGEM-E-ALTERACOES.md).

## Objetivos

- Entender pesos, viés, função de decisão e atualização de parâmetros.
- Implementar o algoritmo de Rosenblatt com NumPy.
- Classificar entradas das portas lógicas **AND** e **OR**.
- Ilustrar visualmente as fronteiras de decisão e os erros por época.
- Demonstrar por que **XOR não é linearmente separável**.
- Validar o comportamento do código com testes.

## Como funciona

O Perceptron calcula uma pontuação linear:

\[
z = \mathbf{w}^{T}\mathbf{x} + b
\]

A classe prevista é `+1` quando `z >= 0`, ou `-1` caso contrário. Durante o treinamento, a cada amostra com margem não positiva, atualiza:

\[
\mathbf{w} \leftarrow \mathbf{w} + \eta y\mathbf{x}, \quad b \leftarrow b + \eta y
\]

Aqui, `η` é a taxa de aprendizado, `y` é o rótulo esperado, `w` são os pesos e `b` o viés. O treinamento para quando uma época termina sem atualizações ou quando chega ao limite de épocas.

**Limitação:** um único Perceptron só representa classificações linearmente separáveis. XOR é um contraexemplo clássico.

## Estrutura

```text
perceptron-do-zero/
├── Perceptron.ipynb             # Notebook com experimentos demonstrativos
├── src/
│   ├── __init__.py
│   └── perceptron.py           # Algoritmo com NumPy
├── tests/
│   └── test_perceptron.py      # Testes automatizados
├── original/
│   └── Perceptron-original.ipynb
├── docs/
│   └── ORIGEM-E-ALTERACOES.md
├── .gitignore
├── LICENSE
├── requirements.txt
├── GUIA-GITHUB.md
└── README.md
```

## Tecnologias

- **Python 3.10+**
- **NumPy**, para cálculos vetoriais
- **Matplotlib**, para gráficos
- **Jupyter Notebook / Google Colab**, para executar os exemplos
- **unittest**, para testes automatizados

O notebook original tinha metadados do **Google Colab**. Esta versão pode ser executada localmente ou no Colab, desde que os arquivos de `src/` estejam disponíveis junto ao notebook.

## Instalação e execução local

Com Python instalado, clone o repositório e entre na pasta:

```bash
git clone https://github.com/CypherGuedes/perceptron-do-zero.git
cd perceptron-do-zero
python -m venv .venv
```

Ative o ambiente virtual:

```bash
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS / Linux
source .venv/bin/activate
```

Instale as dependências:

```bash
python -m pip install -r requirements.txt
```

Abra o notebook:

```bash
jupyter notebook Perceptron.ipynb
```

Execute as células em ordem, a partir da raiz do projeto.

### Execução no Google Colab

Abra o notebook no Colab pelo GitHub ou envie-o manualmente. **Importante:** também é necessário disponibilizar `src/perceptron.py` e `src/__init__.py`; a opção mais simples é clonar o repositório no Colab, entrar na pasta e então executar o notebook. Um notebook aberto isoladamente não encontra automaticamente os módulos em `src`.

## Usando o classificador em Python

```python
import numpy as np
from src.perceptron import Perceptron

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([-1, -1, -1, 1])  # Porta AND

modelo = Perceptron(taxa_aprendizado=0.1, epocas=100)
modelo.ajustar(X, y)

print("Previsões:", modelo.prever(X))
print("Acurácia no treino:", modelo.acuracia(X, y))
print("Erros por época:", modelo.erros_por_epoca_)
```

Execute o exemplo na pasta raiz do projeto com o Python configurado.

## Experimentos incluídos

| Experimento | Objetivo | Resultado esperado |
|---|---|---|
| Porta AND | Aprender uma separação linear | Classificar corretamente os quatro exemplos de treino |
| Porta OR | Aprender outra separação linear | Classificar corretamente os quatro exemplos de treino |
| Porta XOR | Examinar um problema não linear | Mostrar que uma reta não separa todos os exemplos |
| Erros por época | Acompanhar as atualizações | Visualizar o processo de treinamento |
| Fronteira de decisão | Interpretar as previsões | Comparar regiões previstas como -1 e +1 |

Os exemplos são pequenos e determinísticos. **Acurácia de treino não é acurácia de teste**: este projeto não apresenta avaliação de generalização nem resultados em dados externos.

## Testes

Na pasta raiz, execute:

```bash
python -m unittest discover -s tests -v
```

Os testes verificam as portas AND e OR, convergência nos exemplos, validações de parâmetros e tratamento de previsões antes do treinamento.

## Possíveis extensões

- Comparar diferentes taxas de aprendizado e números máximos de épocas.
- Avaliar dados reais linearmente separáveis com conjuntos distintos de treino e teste.
- Comparar o Perceptron com regressão logística e redes com camadas ocultas.
- Explorar classificadores multiclasse e normalização de atributos.

## Licença

Distribuído sob a [Licença MIT](LICENSE). A documentação de origem e das alterações está em [`docs/ORIGEM-E-ALTERACOES.md`](docs/ORIGEM-E-ALTERACOES.md).
