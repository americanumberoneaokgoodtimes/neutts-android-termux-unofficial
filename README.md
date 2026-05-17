# NeuTTS Android (Termux Unofficial)

This is an unofficial, community-maintained repository for running **Neuphonic's NeuTTS** (Text-to-Speech) and **Llama.cpp** (LLM) locally on Android devices via [Termux](https://termux.dev/).

This build includes custom workarounds for hardware-accelerated inference on Snapdragon-based devices, enabling the use of the NPU (Hexagon DSP) and Adreno GPU.

## Features
* **NeuTTS Voice Cloning:** Local inference for voice synthesis.
* **Gradio Web Interface:** A unified dashboard to handle voice synthesis and LLM chat.
* **Hardware Acceleration:** Native NPU (Hexagon) and GPU offloading support for Llama models.
* **Termux Optimized:** Fixes for common Python dependency issues (librosa, llvmlite, sentencepiece, soxr).

## Prerequisites
1. Install [Termux](https://termux.dev/).
2. Run `pkg upgrade`.
3. Install necessary build tools: `pkg install clang python llvm cmake git`.

## Quick Start Guide

### 1. Installation
Clone the repository:
```bash
git clone https://github.com/americanumberoneaokgoodtimes/neutts-android-termux-unofficial
cd neutts-android-termux-unofficial
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Configuration
We have included a detailed setup guide for specific environment issues:
[**Read the full Termux Setup Guide**](termux_setup_guide.md)

### 3. Launch the Control Center
```bash
python3 app.py
```
Open **http://0.0.0.0:7860** in your Android browser to access the web interface.

---

## Hardware Acceleration (Snapdragon 8 Gen 5 / Elite)
This repository leverages the Qualcomm NPU (Hexagon DSP). Ensure your binary is pointed at the correct HTP drivers (as documented in the setup guide) to offload model layers. 

## Known Workarounds Applied
* **Librosa Bypass:** Removed in favor of `torchaudio` to avoid `numba` circular import panics.
* **llvmlite:** Uses the pre-compiled `pkg install python-llvmlite` to avoid LLVM 21 API mismatch issues.
* **soxr:** Compiled manually from source to handle Android's specific linker requirements for Python 3.13.

---
*Disclaimer: This repository is a community-maintained unofficial port and is not affiliated with Neuphonic.*
