# 🐶🐱 Classificador de Cães e Gatos

Projeto desenvolvido em **C# (.NET 9)** para a disciplina de Inteligência Artificial.

O projeto contempla duas etapas:

- **Etapa 2:** Desenvolvimento de uma Rede Neural Convolucional (CNN) utilizando **TorchSharp**.
- **Etapa 3:** Desenvolvimento de um modelo de **Transfer Learning** utilizando **ML.NET** e **ResNetV2-50**.

---

# Tecnologias Utilizadas

- .NET 9
- C#
- ASP.NET Core Web API
- TorchSharp
- ML.NET
- Microsoft.ML.Vision
- TensorFlow (backend do ML.NET)
- Swagger

---

# Estrutura do Projeto

```
ClassificadorCaesGatos
│
├── ClassificadorCaesGatos.Api
│
│   ├── Controllers
│   │      InferenciaController
│   │      TreinamentoController
│   │      TransferLearningController
│   │
│   ├── modelos
│   │      modelo.pt
│   │      modelo-transfer-learning.zip
│   │
│   └── Program.cs
│
└── ClassificadorCaesGatos.IA
    │
    ├── Avaliacao
    ├── Configuracoes
    ├── Dados
    ├── Dtos
    ├── Inferencia
    ├── Interfaces
    ├── RedeNeural
    ├── TransferLearning
    └── Treinamento
```

---

# Dataset

O projeto utiliza o dataset organizado da seguinte forma:

```
dataset/

├── treino
│   ├── cat
│   └── dog
│
├── validacao
│   ├── cat
│   └── dog
│
└── teste
    ├── cat
    └── dog
```

Também são aceitos os nomes em inglês:

```
train
validation
test
```

Cada pasta deve conter duas subpastas:

```
cat
dog
```

---

# Etapa 2 - CNN

Nesta etapa foi desenvolvida manualmente uma Rede Neural Convolucional utilizando TorchSharp.

A arquitetura possui:

- Camadas Convolucionais
- Camadas ReLU
- Camadas MaxPooling
- Camadas Fully Connected
- Softmax
- Cross Entropy Loss
- Adam Optimizer

Também foram implementadas técnicas de Data Augmentation durante o treinamento.

As métricas calculadas foram:

- Loss
- Accuracy
- Precisão
- Recall
- F1 Score

Ao final do treinamento o modelo é salvo em:

```
modelos/modelo.pt
```

---

# Etapa 3 - Transfer Learning

Foi desenvolvido um segundo treinamento utilizando Transfer Learning.

Foi utilizada a arquitetura:

```
ResNetV2-50
```

através do ML.NET.

Nesta etapa foram utilizados:

- TensorFlow
- Microsoft.ML.Vision
- ResNetV2-50 pré-treinada
- Fine Tuning

O modelo treinado é salvo em:

```
modelos/modelo-transfer-learning.zip
```

---

# Como Executar

## 1) Restaurar os pacotes

```
dotnet restore
```

---

## 2) Executar o projeto

```
dotnet run
```

ou abrir a solução no Visual Studio e executar normalmente.

---

## 3) Abrir o Swagger

```
https://localhost:7068/swagger
```

---

# Endpoints

---

## Treinamento CNN

```
POST
/api/treinamento/iniciar
```

Parâmetros:

```
épocas
batchSize
learningRate
```

Exemplo:

```
épocas = 20

batchSize = 32

learningRate = 0.001
```

Resposta:

```
Treinamento iniciado
Modelo salvo
Métricas calculadas
```

---

## Predição CNN

```
POST
/api/inferencia/predizer
```

Enviar:

```
Imagem
```

Resposta:

```json
{
    "classe": "gato",
    "confianca": 99.97
}
```

---

## Treinamento Transfer Learning

```
POST
/api/transfer-learning/treinar
```

Parâmetros:

```
epocas
batchSize
learningRate
```

Exemplo:

```
epocas = 10
batchSize = 16
learningRate = 0.01
```

Resposta:

```json
{
  "success": true,
  "mensagem": "Transfer Learning concluído com sucesso.",
  "arquitetura": "ResNetV2-50",
  "treino": 300,
  "validacao": 100,
  "teste": 100,
  "epocas": 10,
  "batchSize": 16,
  "learningRate": 0.01,
  "acuraciaTeste": 1
}
```

---

## Predição Transfer Learning

```
POST
/api/transfer-learning/predizer
```

Enviar:

```
Imagem
```

Resposta:

```json
{
    "classe":"gato",
    "confianca":99.97,
    "scores":[
        0.0004,
        0.9996
    ]
}
```

---

# Resultados Obtidos

## CNN (TorchSharp)

Foram avaliadas as métricas durante o treinamento:

- Loss
- Accuracy
- Precisão
- Recall
- F1 Score

O modelo treinado foi salvo em:

```
modelos/modelo.pt
```

---

## Transfer Learning

Arquitetura utilizada:

```
ResNetV2-50
```

Resultados obtidos:

- Treino: 300 imagens
- Validação: 100 imagens
- Teste: 100 imagens

Resultado final:

```
Acurácia de Teste

100%
```

Também foram realizados testes com imagens externas ao dataset, demonstrando o funcionamento da inferência em novas imagens.

---

# Observações

O classificador foi treinado **exclusivamente** para duas classes:

- Gato
- Cachorro

Dessa forma, caso seja enviada uma imagem pertencente a outra categoria (como elefante, cavalo, carro ou pessoa), o modelo obrigatoriamente classificará como "gato" ou "cachorro", atribuindo a classe com maior probabilidade.

Esse comportamento é esperado em classificadores supervisionados treinados para apenas duas classes.

---

# Autor

Projeto desenvolvido para a disciplina de Inteligência Artificial utilizando:

- C#
- ASP.NET Core
- TorchSharp
- ML.NET
- TensorFlow
