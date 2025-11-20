# 🌈 PolyglotLab – Smart Translator & Learning Studio

**Live Demo (Hugging Face Space):** https://huggingface.co/spaces/singhalamaan116/PolyglotLab-Translator

**Tech:** Gradio · Hugging Face Transformers · MarianMT · Flan-T5

PolyglotLab is a feature-rich, intelligent translation playground designed to help users translate, learn, and explore languages in a more interactive way than traditional tools.

Unlike simple translator demos, PolyglotLab focuses on:
- 🎭 **Tone-aware translations**  
- 🧩 **Domain-specific language hints**  
- 🔁 **Back-translation meaning-preservation checks**  
- 🧑‍🏫 **AI-powered feedback for learners**  
- 🌍 **Multiple language directions**

Built using Hugging Face models and a modern Gradio interface.

---

## ✨ Key Features

### 🌐 **Smart Translation**
Translate between:
- English  
- French  
- German  
- Spanish  
- Swedish  

With additional controls:
- **Tone:** Neutral · Formal · Informal · Simplified  
- **Domains:** General · Business · Technical · Casual  

### 🔎 **AI Explanation**
See *why* the translation looks the way it does:
- Word choice  
- Tone  
- Grammar  
- Style differences  
- Constraints based on hints  

### 🔁 **Back-Translation Checker**
Validate meaning:
1. Translate *source → target*  
2. Automatically translate *target → source*  
3. Compare results  

Great for spotting ambiguity or information loss.

### 📚 **Learning Mode (Unique Feature)**
Paste your own translation → receive:
- Professional feedback  
- Corrections  
- Suggestions  
- Encouragement  
- Model’s reference translation  

Perfect for students and language enthusiasts.

---

## 🧠 **Models Used**

### Translation Models (MarianMT)
- `Helsinki-NLP/opus-mt-en-fr`
- `Helsinki-NLP/opus-mt-fr-en`
- `Helsinki-NLP/opus-mt-en-de`
- `Helsinki-NLP/opus-mt-de-en`
- `Helsinki-NLP/opus-mt-en-es`
- `Helsinki-NLP/opus-mt-es-en`
- `Helsinki-NLP/opus-mt-en-sv`
- `Helsinki-NLP/opus-mt-sv-en`

### Explanation / Feedback Model
- `google/flan-t5-small`

All models load dynamically to keep the Space fast and lightweight.

