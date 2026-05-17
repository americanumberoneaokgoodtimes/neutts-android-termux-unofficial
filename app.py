import gradio as gr
from neutts import NeuTTS
import os
import soundfile as sf
import torch
import warnings

# Suppress annoying warnings
warnings.filterwarnings("ignore")

# Force environmental workarounds for Android/Termux if needed
os.environ["OPENBLAS_CORETYPE"] = "ARMV8"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

print("Loading NeuTTS Model...")
tts = NeuTTS(
    backbone_repo="neuphonic/neutts-nano-q8-gguf",
    backbone_device="cpu",
    codec_repo="neuphonic/neucodec",
    codec_device="cpu",
    language="en-us"
)
print("Model loaded successfully!")

def synthesize(ref_audio, ref_text, target_text):
    if not ref_audio or not ref_text or not target_text:
        return None, "Please provide all inputs."
    
    try:
        print(f"Encoding reference audio: {ref_audio}")
        ref_codes = tts.encode_reference(ref_audio)
        
        print(f"Synthesizing target text: {target_text}")
        wav = tts.infer(target_text, ref_codes, ref_text)
        
        output_path = "gradio_output.wav"
        sf.write(output_path, wav, 24000)
        return output_path, "Synthesis successful!"
    except Exception as e:
        return None, f"Error during synthesis: {str(e)}"

# Define Gradio Interface
with gr.Blocks(title="NeuTTS Voice Cloning") as demo:
    gr.Markdown("# 🗣️ NeuTTS Voice Cloning Interface")
    gr.Markdown("Clone a voice using a reference audio file and its transcript, then synthesize new text!")
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 1. Reference Data")
            ref_audio_input = gr.Audio(type="filepath", label="Upload Reference Audio (wav/mp3)")
            ref_text_input = gr.Textbox(label="Reference Transcript", placeholder="Enter the exact words spoken in the reference audio...", lines=3)
            
            gr.Markdown("### 2. Target Generation")
            target_text_input = gr.Textbox(label="Target Text", placeholder="Enter the text you want to synthesize in the cloned voice...", lines=4)
            
            generate_btn = gr.Button("Generate Voice", variant="primary")
            
        with gr.Column():
            gr.Markdown("### 3. Output")
            audio_output = gr.Audio(label="Generated Voice")
            status_output = gr.Textbox(label="Status", interactive=False)

    generate_btn.click(
        fn=synthesize,
        inputs=[ref_audio_input, ref_text_input, target_text_input],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
