# voice_chat.py
# Interfaz web con micrófono para tu agente "chatgpt_like_agent.py"
# - Transcribe audio (STT) con OpenAI (whisper-1 por defecto)
# - Responde con tu agente
# - Convierte la respuesta a voz (TTS) y la reproduce en el navegador (opcional)

import os
import tempfile
from openai import OpenAI
import gradio as gr

# Importa tu agente (mismo folder)
import ai_agent as brain

# ======= Config por defecto =======
DEFAULT_STT_MODEL = "whisper-1"              # o "gpt-4o-mini-transcribe" si tu cuenta lo tiene
DEFAULT_TTS_MODEL = "gpt-4o-mini-tts"        # alternativa: "gpt-4o-audio-preview"
DEFAULT_TTS_VOICE = "alloy"                  # alternativas: "aria", "verse", "luna"
MAX_HISTORY = 30

# ======= Inicialización =======
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Falta OPENAI_API_KEY en tu entorno.")
client = OpenAI(api_key=api_key)
agent = brain.ChatLikeAgent()

# ======= Core =======
def chat_with_voice(user_text, user_audio_path, stt_model, tts_model, tts_voice, speak_always):
    """
    user_text: texto escrito (o None)
    user_audio_path: ruta a archivo (gradio type='filepath') grabado por el micrófono (o None)
    stt_model, tts_model, tts_voice: selección de la UI
    speak_always: bool, si True genera y devuelve audio TTS
    Retorna: (historial_messages, ruta_mp3_tts_o_None)
             historial_messages: lista de dicts [{"role":"user|assistant","content":"..."}]
    """
    # 1) Si llegó audio, transcribir
    transcript = None
    if user_audio_path:
        try:
            with open(user_audio_path, "rb") as f:
                tr = client.audio.transcriptions.create(
                    model=stt_model or DEFAULT_STT_MODEL,
                    file=f
                )
            transcript = (tr.text or "").strip()
        except Exception:
            transcript = None

    # 2) Determinar el texto de entrada
    text_in = (user_text or "").strip()
    if not text_in and transcript:
        text_in = transcript
    if not text_in:
        return [], None  # PARA type="messages": debe ser lista de mensajes, no Chatbot.update()

    # 3) Llamar a tu agente
    bot_reply = agent.chat(text_in)

    # 4) TTS (si speak_always)
    tts_path = None
    if speak_always:
        try:
            fd, tts_path = tempfile.mkstemp(suffix=".mp3")
            os.close(fd)
            with client.audio.speech.with_streaming_response.create(
                model=tts_model or DEFAULT_TTS_MODEL,
                voice=tts_voice or DEFAULT_TTS_VOICE,
                input=bot_reply,
                format="mp3",
            ) as resp:
                resp.stream_to_file(tts_path)
        except Exception:
            tts_path = None  # si falla TTS, devolvemos solo texto

    # 5) Construir historial en formato "messages"
    messages = []
    turns = agent.mem.turns[-MAX_HISTORY:]
    for it in turns:
        if it.role in ("user", "assistant"):
            messages.append({"role": it.role, "content": it.text})

    return messages, tts_path

# ======= UI de Gradio =======
with gr.Blocks(title="Mi IA con Voz", theme=gr.themes.Soft()) as demo:
    gr.Markdown("## 🗣️ IA con voz (web + TTS + STT)\nHabla o escribe. La IA responde con voz y texto.")

    # IMPORTANTE: type="messages" -> espera [{"role":..., "content":...}, ...]
    chatbot = gr.Chatbot(type="messages", height=420, show_copy_button=True)

    with gr.Accordion("Ajustes de voz", open=False):
        with gr.Row():
            stt_model = gr.Dropdown(
                label="Modelo STT (transcripción)",
                choices=["whisper-1", "gpt-4o-mini-transcribe"],
                value=DEFAULT_STT_MODEL
            )
            tts_model = gr.Dropdown(
                label="Modelo TTS (texto a voz)",
                choices=["gpt-4o-mini-tts", "gpt-4o-audio-preview"],
                value=DEFAULT_TTS_MODEL
            )
            tts_voice = gr.Dropdown(
                label="Voz",
                choices=["alloy","aria","verse","luna"],
                value=DEFAULT_TTS_VOICE
            )
            speak_always = gr.Checkbox(value=True, label="Hablar siempre (TTS)")

    text_in = gr.Textbox(label="Escribe aquí (opcional)", placeholder="Di o escribe tu mensaje…", lines=2)
    audio_in = gr.Audio(
        sources=["microphone"],
        type="filepath",   # nos pasa ruta de archivo (estable en Windows)
        label="Graba tu voz (opcional)"
    )

    with gr.Row():
        send_btn = gr.Button("Enviar", variant="primary")
        clear_btn = gr.Button("Limpiar chat")

    audio_out = gr.Audio(label="Respuesta en voz", autoplay=True)

    def handle_send(t, a, stt, tts, voice, speak):
        msgs, tts_path = chat_with_voice(t, a, stt, tts, voice, speak)
        # Devolver: historial messages, limpiar inputs, y el audio
        return msgs, "", None, tts_path

    send_btn.click(
        handle_send,
        inputs=[text_in, audio_in, stt_model, tts_model, tts_voice, speak_always],
        outputs=[chatbot, text_in, audio_in, audio_out]
    )

    def handle_clear():
        return [], "", None, None

    clear_btn.click(handle_clear, outputs=[chatbot, text_in, audio_in, audio_out])

if __name__ == "__main__":
    # Cambia a share=True si quieres URL pública temporal
    demo.launch(share=False)
