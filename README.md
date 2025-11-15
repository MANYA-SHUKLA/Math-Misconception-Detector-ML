SLIDE 1 — Title Slide
Math Misconception Detector – AI/ML Project

An AI-powered system to detect and diagnose mathematical misconceptions

Full-stack architecture: Frontend + Backend + Python ML Core

Built using Machine Learning, NLP & Modern Web Technologies

SLIDE 2 — Repository Overview
Project Summary

Purpose: Identify math misconceptions in student responses

Status: Public GitHub repo (~11 days old, 108 KB)

Primary Language: Python (ML core)

Other Tech: Next.js + Node.js

SLIDE 3 — Architecture
Three-tier Full Stack Setup

Frontend: Next.js (React + TS)

Backend: Node.js (Express API Gateway)

ML Engine: FastAPI (Python – NLP + ML models)

Communicates via REST APIs

SLIDE 4 — Backend (Node.js)
Role of Backend

Middleware between frontend ↔ ML engine

Handles routing, security, environment config, logging

Tech Stack

Express, CORS, Helmet, Morgan, Dotenv

Folders

controllers/, middleware/, routes/, utils/

SLIDE 5 — Frontend (Next.js)
Purpose

User interface for entering student statements

Displays misconceptions + interventions

Tech Stack

Next.js 16, React 19, TypeScript

Tailwind CSS for UI

ESLint + PostCSS

Structure

src/app/, src/components/, src/utils/, public/

SLIDE 6 — Python ML Engine (FastAPI)
Core Engine for Misconception Detection

FastAPI server for ML inference

Uses NLP + Transformers + Embeddings

Major Libraries

Transformers, Sentence Transformers

NLTK, Spacy, Scikit-learn

PyTorch, Pandas, NumPy

SLIDE 7 — Core ML Components
Key Classes in main.py

AdvancedMathMisconceptionLoader
Loads & compiles misconception datasets

AdvancedNLPAnalyzer
Performs NLP: tokenizing, POS tagging, pattern detection

MLMisconceptionDetector
Uses embeddings (all-mpnet-base-v2) + similarity scoring

SLIDE 8 — Major Features

Semantic + pattern-based detection

Personalized learning interventions

Grade-level + topic-specific mapping

Async data loading with caching

Covers arithmetic → algebra → geometry → statistics

SLIDE 9 — Workflow (How It Works)

Data loaded from curated sources

User submits math statement

NLP processing + pattern extraction

ML model computes similarities

Detects misconception + confidence

Returns detailed feedback & intervention steps

SLIDE 10 — Key Strengths

Comprehensive math misconception coverage

Modern, scalable AI architecture

High accuracy using embeddings

Useful for teachers, students, and edu-tech apps
