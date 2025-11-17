#!./.venv/bin/python3

# imports
import sys
from openai import OpenAI
import os
from dotenv import load_dotenv
from IPython.display import Markdown, display, update_display
import base64
from io import BytesIO
import requests
import gradio as gr

# load environment variables
load_dotenv(override=True)
api_key = os.getenv('OPENAI_API_KEY')

# constants
MODEL_GPT = 'gpt-4o-mini'
MODEL_LLAMA = 'llama3.2'
ollama_url = "http://localhost:11434/v1"

# set up environment
openai = OpenAI()
ollama = OpenAI(api_key="ollama", base_url=ollama_url)

# helper functions
def system_prompt(selected_model):
    return f"""
    You are a tech expert and know every coding language, and can give 
    nice, detailed and simple explanations for the given questions.
    Introduce yourself by saying which model you are every time you answer. For example, this is {selected_model}.  
    """

def talker(message):
    response = openai.audio.speech.create(
        model="gpt-4o-mini-tts",
        voice="onyx",
        input=message)

    audio_stream = BytesIO(response.content)
    output_filename = "output_audio.mp3"
    with open(output_filename, "wb") as f:
        f.write(audio_stream.read())

    return output_filename  # Return the file path instead of displaying

def listener(audio_file):
    with open(audio_file, "rb") as audio:
        transcript = openai.audio.transcriptions.create(
            model="whisper-1",
            file=audio
        )
    return transcript.text

# main callback function
def chat(history, selected_model):
    history = [{"role":h["role"], "content":h["content"]} for h in history]
    messages = [{"role": "system", "content": system_prompt(selected_model)}] + history

    if selected_model == "GPT-4o-mini":
        stream = openai.chat.completions.create(model=MODEL_GPT, messages=messages, stream=True)
        response = ""
        
        for chunk in stream:
            try:
                response += chunk.choices[0].delta.content or ''
                updated_history = history + [{"role": "assistant", "content": response}]
                yield updated_history, None  # Yield None for audio during streaming
            except Exception as e:
                print(f"Streaming error: {e}")
                yield "Sorry, there was an error processing your request."

        if response:
            audio_file = talker(response)
            yield updated_history, audio_file  # Yield the final audio file

    elif selected_model == "Llama3.2":
        stream = ollama.chat.completions.create(model=MODEL_LLAMA, messages=messages, stream=True)        
        response = ""
        
        for chunk in stream:
            try:
                response += chunk.choices[0].delta.content or ''
                updated_history = history + [{"role": "assistant", "content": response}]
                yield updated_history, None  # Yield None for audio during streaming
            except Exception as e:
                print(f"Streaming error: {e}")
                yield "Sorry, there was an error processing your request."

        if response:
            audio_file = talker(response)
            yield updated_history, audio_file  # Yield the final audio file

"""
main.py: Entry point for the Python module.
"""
def main():
    with gr.Blocks() as ui:
        gr.Markdown("## LLM TECH EXPERT")
        gr.Markdown("**Select your preferred AI model:**")
        
        model_dropdown = gr.Dropdown(
            choices=["GPT-4o-mini", "Llama3.2"], 
            value="GPT-4o-mini",  # default selection
            label="Choose Model"
        )

        with gr.Row():
            chatbot = gr.Chatbot(height=200, type="messages")
        with gr.Row():
            audio_output = gr.Audio(autoplay=True)
        with gr.Row():
            entry = gr.Textbox(label="Ask a tech question:")
        with gr.Row():
            # Audio input for voice messages
            audio_input = gr.Audio(
                sources=["microphone", "upload"], 
                type="filepath", 
                label="🎙️ Voice Message"
            )
        with gr.Row():
            voice_submit = gr.Button("Send Voice Message", variant="secondary")
            clear = gr.Button("Clear")

        def do_entry(message, history):
            history += [{"role":"user", "content":message}]
            return "", history

        def process_voice_input(audio_file):
            """Convert voice to text and put it in the text box"""
            if audio_file is not None:
                transcribed_text = listener(audio_file)
                if transcribed_text and not transcribed_text.startswith("Error"):
                    return transcribed_text
            return ""
        
        entry.submit(do_entry, inputs=[entry, chatbot], outputs=[entry, chatbot]).then(
            chat, inputs=[chatbot, model_dropdown], outputs=[chatbot, audio_output]
        )

        voice_submit.click(
            process_voice_input,
            inputs=[audio_input],
            outputs=[entry]
        ).then(
            do_entry,
            inputs=[entry, chatbot],
            outputs=[entry, chatbot]
        ).then(
            chat,
            inputs=[chatbot, model_dropdown],
            outputs=[chatbot, audio_output]
        )

        clear.click(lambda: None, inputs=None, outputs=chatbot, queue=False)

    ui.launch(inbrowser=True)

if __name__ == "__main__":
    # Ensures the script runs only when executed directly
    sys.exit(main())