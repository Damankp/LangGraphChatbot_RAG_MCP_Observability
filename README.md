# 📘 LangGraph Multi-Utility PDF Chatbot

## 🔑 Key Concepts Used
- **LangGraph** (stateful agent workflow, tool routing, checkpointing)  
- **RAG with FAISS** (PDF embeddings + vector search)  
- **OpenAI Embeddings** + **Recursive Text Splitter**  
- **Tool Calling** (RAG, Web Search, Stock Price API, Calculator)  
- **Multi-Thread Memory** using **SQLite checkpointing**  
- **Streamlit Chat UI** with real-time response streaming (as a typewriter effect)

---

## ✨ Overview
A Streamlit-based AI assistant that allows users to upload PDFs, ask questions using Retrieval-Augmented Generation (RAG), and perform actions using built-in tools. Each chat thread maintains its own memory, retriever, and history using LangGraph’s stateful architecture.

---
