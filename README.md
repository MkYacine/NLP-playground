# NLP Playground 🧠

Welcome to my NLP Playground! This repository documents my self-taught journey in Natural Language Processing and Deep Learning. Here, I implement research papers, conduct curiosity-driven experiments, and deploy models following MLOps industry standards.  
The ressources for these experiments are limited. My main goal here is to learn.  
  
## Projects Overview

### 1. LLMs From Scratch ✅
A comprehensive implementation of modern Language Models from the ground up, guided by Sebastian Raschka's "Building LLMs from Scratch". This project covers:

- Text preprocessing pipelines
- Attention mechanism implementation and visualization
- Complete transformer architecture
- GPT-2 model implementation
- Pretraining phase development
- Finetuning for:
  - Text classification
  - Instruction following

**References:**
- [Attention Is All You Need](https://arxiv.org/abs/1706.03762)
- [InstructGPT: Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155)
- [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch)

### 2. Financial NER with FiNER-ORD ✅
Implementation and experimentation with the Financial NER Open Research Dataset (FiNER-ORD), exploring various approaches to Named Entity Recognition in the financial domain.  
Main notebook can be seen [here](https://github.com/MkYacine/NLP-playground/blob/main/FiNER/FiNER_comparison.ipynb)

- BERT finetuning on FiNER-ORD
- Model deployment on AWS SageMaker using structured jobs (for reproducability and practice purposes)
- Closed-source LLM
- Base GLiNER model
- Document findings and compare performance
- Finetuned GLiNER model 🚧

**References:**
- [FiNER-ORD: Building a High-Quality English Financial NER Open Research Dataset](https://arxiv.org/abs/2302.11157)
- [GLiNER: Generalist Model for Named Entity Recognition using Bidirectional Transformer](https://arxiv.org/abs/2311.08526)

### 3. RAG 🚧
Implementation and experimentation with retrieval augmented generation, using the huggingface documentation as a knowledge base, trying different approaches in the RAG pipeline.

- Embedding and Retrieval: Test different strategies for chunking, embedding, retrieval, and reranking
- Generator: Test different generator models, prompts, and verification strategies.

**References:**
- [HuggingFace Guide: Advanced RAG on Hugging Face documentation using LangChain](https://huggingface.co/learn/cookbook/en/advanced_rag)
- [ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488)
- [Creating a LLM-as-a-Judge That Drives Business Results](https://hamel.dev/blog/posts/llm-judge/#the-problem-ai-teams-are-drowning-in-data)
- [A curated knowledge base of real-world LLMOps implementations, with detailed summaries and technical notes.](https://www.zenml.io/llmops-database)
- [Anthropic's Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)

### More to come in the future  


## Legend
- ✅ Completed
- 🚧 In Progress
- 📋 Planned