import gradio as gr
from transformers import pipeline

# -----------------------
# 1. Language + model config
# -----------------------

LANG_CODES = {
    "English": "en",
    "French": "fr",
    "German": "de",
    "Spanish": "es",
    "Swedish": "sv",
}

# Map (src_lang_code, tgt_lang_code) -> MarianMT model
MODEL_MAP = {
    ("en", "fr"): "Helsinki-NLP/opus-mt-en-fr",
    ("fr", "en"): "Helsinki-NLP/opus-mt-fr-en",
    ("en", "de"): "Helsinki-NLP/opus-mt-en-de",
    ("de", "en"): "Helsinki-NLP/opus-mt-de-en",
    ("en", "es"): "Helsinki-NLP/opus-mt-en-es",
    ("es", "en"): "Helsinki-NLP/opus-mt-es-en",
    ("en", "sv"): "Helsinki-NLP/opus-mt-en-sv",
    ("sv", "en"): "Helsinki-NLP/opus-mt-sv-en",
}

# Lazy-loaded translation pipelines
_translation_pipelines = {}

# One small LLM for explanations / feedback
explain_llm = pipeline("text2text-generation", model="google/flan-t5-small")


def get_translation_pipeline(src_code: str, tgt_code: str):
    """
    Returns a transformers pipeline for a given language pair, loading it lazily.
    """
    key = (src_code, tgt_code)
    if key not in MODEL_MAP:
        raise ValueError(f"Language pair {src_code}->{tgt_code} not supported yet.")
    if key not in _translation_pipelines:
        model_name = MODEL_MAP[key]
        task = f"translation_{src_code}_to_{tgt_code}"
        _translation_pipelines[key] = pipeline(task, model=model_name)
    return _translation_pipelines[key]


# -----------------------
# 2. Core translation logic
# -----------------------

def _apply_style_hints(text: str, tone: str, domain: str, tgt_lang: str) -> str:
    """
    MarianMT isn't instruction-tuned, but we can still stuff a hint into the input.
    It won't be perfect, but conceptually shows tone/domain-aware translation.
    """
    hints = []
    if domain != "General":
        hints.append(f"{domain} context")
    if tone != "Neutral":
        hints.append(f"{tone} tone")

    if hints:
        hint_str = ", ".join(hints)
        # Just prepend some natural-language hints in English.
        styled = f"[{hint_str} in {tgt_lang}] {text}"
        return styled
    return text


def translate_text(text: str, src_lang: str, tgt_lang: str, tone: str, domain: str):
    """
    Main translation function for the UI.
    """
    text = (text or "").strip()
    if not text:
        return "Please enter some text to translate."

    if src_lang == tgt_lang:
        return text  # trivial case

    src_code = LANG_CODES[src_lang]
    tgt_code = LANG_CODES[tgt_lang]

    try:
        translator = get_translation_pipeline(src_code, tgt_code)
    except ValueError as e:
        return str(e)

    styled_input = _apply_style_hints(text, tone, domain, tgt_lang)

    out = translator(styled_input, max_length=512)
    translated = out[0]["translation_text"]
    return translated.strip()


def back_translate(text: str, src_lang: str, tgt_lang: str, tone: str, domain: str):
    """
    Translate from src -> tgt, then back tgt -> src to check meaning preservation.
    """
    text = (text or "").strip()
    if not text:
        return "Please enter some text to translate.", ""

    if src_lang == tgt_lang:
        return text, text

    # First translation: src -> tgt
    forward = translate_text(text, src_lang, tgt_lang, tone, domain)
    # Back translation: tgt -> src (no style hints on the way back)
    backward = translate_text(forward, tgt_lang, src_lang, "Neutral", "General")

    return forward, backward


def explain_translation(src_text: str, translated_text: str, src_lang: str, tgt_lang: str):
    """
    Use Flan-T5 to explain the translation in simple terms.
    """
    src_text = (src_text or "").strip()
    translated_text = (translated_text or "").strip()

    if not src_text or not translated_text:
        return "Provide both the original text and the translation to get an explanation."

    prompt = (
        "You are a helpful language teacher. "
        "Explain this translation to a learner in simple terms. "
        "Mention important word choices, tone, and any interesting grammar.\n\n"
        f"Source language: {src_lang}\n"
        f"Target language: {tgt_lang}\n\n"
        f"Original text:\n{src_text}\n\n"
        f"Translation:\n{translated_text}\n\n"
        "Explanation (in English, 1–2 short paragraphs):"
    )

    out = explain_llm(prompt, max_new_tokens=256, temperature=0.4)
    return out[0]["generated_text"].strip()


def learning_mode_feedback(src_text: str, user_translation: str, src_lang: str, tgt_lang: str):
    """
    Compare user's translation to model translation and give feedback.
    """
    src_text = (src_text or "").strip()
    user_translation = (user_translation or "").strip()

    if not src_text or not user_translation:
        return "Please provide both the original text and your translation."

    # Model's best guess (neutral, general)
    model_translation = translate_text(src_text, src_lang, tgt_lang, "Neutral", "General")

    prompt = (
        "You are a friendly language teacher. Compare the student's translation to the model translation. "
        "Explain what is good, what could be improved, and give 2–4 concrete suggestions. "
        "Be encouraging, not harsh.\n\n"
        f"Source language: {src_lang}\n"
        f"Target language: {tgt_lang}\n\n"
        f"Original text:\n{src_text}\n\n"
        f"Student's translation:\n{user_translation}\n\n"
        f"Model's translation:\n{model_translation}\n\n"
        "Feedback (in English, short and structured):"
    )

    out = explain_llm(prompt, max_new_tokens=320, temperature=0.4)
    feedback = out[0]["generated_text"].strip()

    return f"**Model translation:**\n\n{model_translation}\n\n---\n\n**Feedback:**\n\n{feedback}"


# -----------------------
# 3. Gradio UI
# -----------------------

LANG_CHOICES = list(LANG_CODES.keys())
TONES = ["Neutral", "Formal", "Informal", "Simplified"]
DOMAINS = ["General", "Business", "Technical", "Casual"]

with gr.Blocks(title="PolyglotLab – Smart Translator & Learning Studio") as demo:
    gr.Markdown(
        """
        # 🌈 PolyglotLab – Smart Translator & Learning Studio

        A translation playground built with Hugging Face + Gradio.

        - ✨ Multi-language translation (English, French, German, Spanish, Swedish)  
        - 🎭 Tone hints (neutral, formal, informal, simplified)  
        - 🧩 Domain hints (business, technical, casual)  
        - 🔁 Back-translation checks for meaning  
        - 📚 Learning mode with feedback on *your* translations  
        """
    )

    with gr.Tab("Smart Translate"):
        with gr.Row():
            src_lang_in = gr.Dropdown(LANG_CHOICES, value="English", label="Source language")
            tgt_lang_in = gr.Dropdown(LANG_CHOICES, value="French", label="Target language")

        text_in = gr.Textbox(
            label="Text to translate",
            lines=4,
            placeholder="Type or paste text here...",
        )

        with gr.Row():
            tone_in = gr.Dropdown(TONES, value="Neutral", label="Tone hint")
            domain_in = gr.Dropdown(DOMAINS, value="General", label="Domain / context")

        explain_checkbox = gr.Checkbox(value=True, label="Explain the translation")

        translate_btn = gr.Button("Translate ✨")

        translated_out = gr.Textbox(label="Translation", lines=4)
        explanation_out = gr.Markdown(label="Explanation")

        def translate_and_explain(text, src, tgt, tone, domain, do_explain):
            translation = translate_text(text, src, tgt, tone, domain)
            if not do_explain:
                return translation, ""
            exp = explain_translation(text, translation, src, tgt)
            return translation, exp

        translate_btn.click(
            fn=translate_and_explain,
            inputs=[text_in, src_lang_in, tgt_lang_in, tone_in, domain_in, explain_checkbox],
            outputs=[translated_out, explanation_out],
        )

    with gr.Tab("Back-translation Check"):
        gr.Markdown(
            "Translate from source to target, then back to source to see if the meaning is preserved."
        )

        bt_src_lang = gr.Dropdown(LANG_CHOICES, value="English", label="Source language")
        bt_tgt_lang = gr.Dropdown(LANG_CHOICES, value="German", label="Target language")
        bt_text_in = gr.Textbox(
            label="Original text",
            lines=4,
            placeholder="Type a sentence to test...",
        )

        bt_tone_in = gr.Dropdown(TONES, value="Neutral", label="Tone hint")
        bt_domain_in = gr.Dropdown(DOMAINS, value="General", label="Domain / context")

        bt_btn = gr.Button("Run Back-translation 🔁")

        bt_forward_out = gr.Textbox(label="Forward translation (src → tgt)", lines=4)
        bt_backward_out = gr.Textbox(label="Back-translation (tgt → src)", lines=4)

        bt_btn.click(
            fn=back_translate,
            inputs=[bt_text_in, bt_src_lang, bt_tgt_lang, bt_tone_in, bt_domain_in],
            outputs=[bt_forward_out, bt_backward_out],
        )

    with gr.Tab("Learning Mode"):
        gr.Markdown(
            """
            Paste a sentence and your own translation.  
            The model will show its translation and give you friendly feedback.
            """
        )

        lm_src_lang = gr.Dropdown(LANG_CHOICES, value="English", label="Source language")
        lm_tgt_lang = gr.Dropdown(LANG_CHOICES, value="French", label="Target language")

        lm_src_text = gr.Textbox(
            label="Original text",
            lines=4,
            placeholder="Enter a sentence in the source language...",
        )

        lm_user_translation = gr.Textbox(
            label="Your translation",
            lines=4,
            placeholder="Write your translation here...",
        )

        lm_btn = gr.Button("Get feedback 🧑‍🏫")
        lm_feedback_out = gr.Markdown(label="Feedback")

        lm_btn.click(
            fn=learning_mode_feedback,
            inputs=[lm_src_text, lm_user_translation, lm_src_lang, lm_tgt_lang],
            outputs=lm_feedback_out,
        )

if __name__ == "__main__":
    demo.launch()
