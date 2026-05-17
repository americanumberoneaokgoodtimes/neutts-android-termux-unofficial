import gradio as gr
from neutts import NeuTTS
import os
import soundfile as sf
import torch
import warnings
import subprocess
import requests
import time

warnings.filterwarnings("ignore")
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Configuration paths
MODEL_DIR = "./model/"
LLAMA_SERVER_BIN = "/data/data/com.termux/files/home/llama/binaries/src/llama.cpp/build-android-htp/bin/llama-server"

class AppState:
    def __init__(self):
        self.llama_process = None
        self.tts = None

state = AppState()

def get_models():
    return [f for f in os.listdir(MODEL_DIR) if f.endswith(".gguf")]

def start_llama_server(model_name, n_ctx, n_gpu_layers):
    if state.llama_process:
        state.llama_process.terminate()
    
    model_path = os.path.join(MODEL_DIR, model_name)
    cmd = [
        LLAMA_SERVER_BIN,
        "-m", model_path,
        "-c", str(n_ctx),
        "-ngl", str(n_gpu_layers),
        "--host", "127.0.0.1",
        "--port", "8080"
    ]
    state.llama_process = subprocess.Popen(cmd)
    return f"Llama server started with {model_name}"

def chat_interface(message, history):
    if not state.llama_process:
        return "Llama server not running."
    
    try:
        response = requests.post("http://127.0.0.1:8080/completion", json={"prompt": message, "n_predict": 128})
        return response.json().get("content", "Error generating response.")
    except Exception as e:
        return f"Error: {e}"

def synthesize(ref_audio, ref_text, target_text, backbone, language):
    if not state.tts or state.tts.backbone_repo != backbone:
        state.tts = NeuTTS(backbone_repo=backbone, language=language)
        
    ref_codes = state.tts.encode_reference(ref_audio)
    wav = state.tts.infer(target_text, ref_codes, ref_text)
    output_path = "output.wav"
    sf.write(output_path, wav, 24000)
    return output_path

with gr.Blocks() as demo:
    gr.Markdown("# 🚀 NeuTTS & Llama Control Center")
    
    with gr.Tab("Voice Generator"):
        with gr.Row():
            ref_audio = gr.Audio(type="filepath", label="Ref Audio")
            ref_text = gr.Textbox(label="Ref Transcript")
        backbone = gr.Dropdown(choices=["neuphonic/neutts-nano-q8-gguf", "neuphonic/neutts-air-q4-gguf"], label="Backbone")
        language = gr.Dropdown(choices=["en-us", "fr-fr"], value="en-us", label="Language")
        target_text = gr.Textbox(label="Target Text")
        gen_btn = gr.Button("Synthesize")
        gen_output = gr.Audio()
        gen_btn.click(synthesize, [ref_audio, ref_text, target_text, backbone, language], gen_output)

    with gr.Tab("Llama Chat"):
        model_dropdown = gr.Dropdown(choices=get_models(), label="Model")
        n_ctx = gr.Slider(256, 4096, 512, label="Context Size")
        n_gpu = gr.Slider(0, 99, 99, label="GPU Layers (NPU Offload)")
        start_btn = gr.Button("Start Server")
        start_btn.click(start_llama_server, [model_dropdown, n_ctx, n_gpu])
        gr.ChatInterface(chat_interface)

demo.launch(server_name="0.0.0.0", server_port=7860)
