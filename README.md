📚 CRAG – Corrective Retrieval Augmented Generation System

A fully implemented CRAG (Corrective RAG) pipeline built from scratch using LangChain, LangGraph, FAISS, OpenAI, and Tavily Search.

This project enhances traditional RAG by introducing retrieval evaluation, correction mechanisms, and web fallback search to reduce hallucinations and improve answer reliability.

🚀 Project Overview

Traditional RAG (Retrieval Augmented Generation) generates answers from retrieved documents. However:

Retrieved chunks may be partially relevant

Retrieval may be incorrect

The LLM may hallucinate

Knowledge base may lack complete information

To solve this, this project implements a Corrective RAG (CRAG) pipeline with:

✔ Retrieval Refinement
✔ Retrieval Evaluation using LLM
✔ Automatic Web Search Fallback (Tavily)
✔ Query Rewriting
✔ Multi-step LangGraph pipeline

🧠 System Architecture
Stage 1 — Basic RAG

Load PDFs

Chunk documents

Create embeddings

Store in FAISS vector database

Retrieve relevant chunks

Generate answer using ChatOpenAI

Stage 2 — Retrieval Refinement

Improved retrieval quality by:

Better chunking strategy

Refined similarity search

Cleaner preprocessing

Goal: Reduce noisy or partially relevant retrieval.

Stage 3 — Retrieval Evaluator (Core CRAG Logic)

An LLM-based evaluator checks:

Is retrieved context relevant?

Is it partially correct?

Is it completely incorrect?

Based on evaluation:

Condition	Action
Relevant	Generate final answer
Partially relevant	Refine retrieval
Incorrect	Trigger Web Search
Stage 4 — Web Search Fallback (Tavily)

If retrieval fails:

Use TavilySearchResults

Fetch external web results

Inject into context

Regenerate answer

This ensures:

The model does NOT hallucinate when internal knowledge is insufficient.

Stage 5 — Query Rewrite

If retrieval quality is weak:

LLM rewrites the query

Improves semantic retrieval

Re-runs retrieval pipeline

Final Stage — Complete CRAG Flow (LangGraph)

All components are integrated using LangGraph StateGraph:

START
  ↓
Retrieve Docs
  ↓
Evaluate Retrieval
  ↓
[Relevant?]
   ├── Yes → Generate Answer → END
   └── No
        ↓
   Web Search
        ↓
   Generate Answer → END

   📂 Knowledge Base

The project uses 3 books (PDF format) as training knowledge base for RAG.

These are loaded using:

PyPDFLoader


Then chunked using:

RecursiveCharacterTextSplitter


Embedded using:

OpenAIEmbeddings


Stored in:

FAISS

🛠️ Tech Stack & Libraries
from typing import List, TypedDict, Literal
from pydantic import BaseModel
import re

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv

from langchain_community.tools.tavily_search import TavilySearchResults
