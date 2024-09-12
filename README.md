![Project Status](https://img.shields.io/badge/Project-Completed-brightgreen)

<div style="padding:15px 20px 20px 20px;border-left:3px solid blue;background-color:#e4f0fa;border-radius: 20px;color:#424242;">

## **NLP Language Modeling & Text Simplification**

This project applies advanced NLP techniques to train and fine-tune language models using the wikitext-103 dataset from Wikipedia. It encompasses data preprocessing, model training, and fine-tuning for text simplification tasks.

### **Project Overview**
- **Data Preprocessing:** Development and cleaning of a custom dataset, including vocabulary building and PyTorch dataset creation.
- **Model Training:** Implementation and training of various LSTM and Transformer models.
- **Model Fine-tuning:** Fine-tuning models for text simplification tasks.

### **Data and Models**

- **Dataset:** Utilized the [wikitext-103 dataset](https://huggingface.co/datasets/wikitext), a large collection of high-quality Wikipedia articles.

- **Model Variants:**
  - **LSTM with Token Embeddings Trained from Scratch:** An LSTM model with a trainable token embedding layer initialized randomly and trained alongside the language model.
  - **LSTM with Pre-trained GloVe Embeddings:** An LSTM model using pre-trained GloVe embeddings, which remain frozen during training.
  - **Transformer (DistilGPT2-based):** A Transformer model trained from scratch, following the architecture of DistilGPT2.

### **Fine-tuning Tasks**

- **Custom Encoder-Decoder Model:** Fine-tuned on the text simplification task, which involves compressing text to extract the most important information.
- **Pretrained T5 Model:** Fine-tuned on the same text simplification task, leveraging a pretrained encoder-decoder model.

### **Technical Details**
- Implemented models using Python with libraries such as PyTorch and Hugging Face Transformers.
- Utilized Git LFS for managing large files and model checkpoints.
- Employed GPU acceleration through Colab for efficient training.

### **Codebase File Structure**

```txt
.
├── distilgpt2-wikitext103
│   ├── ...
├── docs
    ├── ...
├── models
    ├── ...
├── tensorboard
    ├── ...
├── README.md
├── data.py
├── modeling.py
├── requirements.txt
├── test_A1.py
├── train.ipynb (main notebook)
└── utils.py
```

</div>