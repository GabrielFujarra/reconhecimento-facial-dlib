# 🧠 Reconhecimento Facial com Dlib

Este projeto implementa um sistema de **reconhecimento facial** utilizando a biblioteca **dlib**, aplicando técnicas modernas de visão computacional para identificar pessoas a partir de imagens.

---

## 🚀 Funcionalidades

* 🔍 Detecção de faces em imagens
* 📍 Extração de **68 pontos faciais (landmarks)**
* 🧬 Geração de **descritores faciais (vetores de 128 dimensões)**
* 📏 Comparação entre faces usando distância euclidiana
* 🧑‍💻 Identificação de indivíduos em imagens de teste

---

## 🧠 Como funciona

O pipeline do projeto segue as seguintes etapas:

1. **Detecção de face**
   Utiliza o detector frontal do dlib para localizar rostos na imagem.

2. **Extração de pontos faciais (landmarks)**
   São detectados 68 pontos importantes do rosto (olhos, nariz, boca, etc).

3. **Geração do descritor facial**
   Cada face é transformada em um vetor numérico de 128 dimensões.

4. **Comparação entre faces**
   A similaridade entre duas faces é medida pela distância euclidiana:

   * Distância menor → maior similaridade
   * Distância maior → pessoas diferentes

5. **Classificação**
   A face é identificada com base no menor valor de distância.

---

## 📁 Estrutura do Projeto

```
ReconhecimentoComDlib/
│
├── src/
│   └── reconhecimento_com_dlib.py
│
├── data/
│   └── yalefaces/
│       ├── train/
│       └── test/
│
├── models/
│   ├── shape_predictor_68_face_landmarks.dat
│   └── dlib_face_recognition_resnet_model_v1.dat
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 📦 Requisitos

Instale as dependências com:

```bash
pip install -r requirements.txt
```

Ou manualmente:

```bash
pip install dlib opencv-python numpy pillow
```

---

## 📥 Download dos Modelos

Os modelos do dlib **não estão incluídos no repositório** devido ao tamanho.

Baixe manualmente:

* 📌 http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
* 📌 http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2

Após baixar:

1. Extraia os arquivos
2. Coloque na pasta:

```
models/
```

---

## 📊 Dataset

O projeto utiliza o dataset **Yale Faces**.

Estrutura esperada:

```
data/yalefaces/
├── train/
└── test/
```

Caso não tenha o dataset, você pode buscar por:

```
Yale Face Database
```

---

## ▶️ Como executar

Execute o script principal:

```bash
python src/reconhecimento_com_dlib.py
```

---

## 🖼️ Saída

O sistema exibirá:

* 🔲 Bounding box da face
* 📍 Pontos faciais
* 🧑 Predição (ID previsto)
* ✅ Valor esperado (ID real)

---

## ⚙️ Parâmetros importantes

```python
confiancaMinima = 0.5
```

* Valores menores → mais rigoroso
* Valores maiores → mais permissivo

---

## 🧪 Tecnologias utilizadas

* Python 🐍
* dlib 🧠
* OpenCV 👁️
* NumPy 📊
* Pillow 🖼️

---

## 📌 Possíveis melhorias

* 🔴 Reconhecimento em tempo real com webcam
* 📈 Uso de classificadores (KNN, SVM)
* 💾 Persistência dos descritores em banco de dados
* 🎯 Ajuste automático do limiar de confiança

---

## 👨‍💻 Autor

Desenvolvido por **Gabriel Fujarra**

---

## ⭐ Considerações

Este projeto foi desenvolvido com fins educacionais, com foco em aprendizado de:

* Visão computacional
* Processamento de imagens
* Machine Learning aplicado

---

Se este projeto te ajudou, considere dar uma ⭐ no repositório!
