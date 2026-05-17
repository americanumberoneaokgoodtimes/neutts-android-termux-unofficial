# Termux / Android Setup Guide for NeuTTS

Running NeuTTS natively on Android via Termux poses unique challenges due to differences in standard Linux build toolchains and platform detections. Here is a guide on how to resolve the main issues encountered during environment setup.

## 1. Environment Variables for Crash Workarounds

OpenBLAS and OpenMP binaries shipped in Python packages or Termux packages can sometimes cause crashes during inference. Add the following to your `~/.bashrc`:

```bash
# Prevents "BLAS : Bad memory unallocation" error during GGUF inference
export OPENBLAS_CORETYPE=ARMV8

# Bypasses OpenMP "multiple copies initialized" crash
export KMP_DUPLICATE_LIB_OK=TRUE
```

## 2. Tokenizers and Transformers Compatibility

Depending on the `transformers` and `tokenizers` versions, you may hit Rust build failures (e.g., PyO3 version mismatches).
To compile modern `tokenizers` smoothly on Python 3.13, you may need to force ABI3 compatibility:

```bash
export PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1
pip install transformers==4.47.0
```

## 3. Hugging Face Hub (hf-xet rustls panic)

The `hf-xet` package attempts to use a Rust-based platform verifier that is known to panic on Android (`Expect rustls-platform-verifier to be initialized`).
The easiest fix is to completely uninstall `hf-xet` and force standard HTTP downloads:

```bash
pip uninstall -y hf-xet
```

## 4. `llama-cpp-python` Platform Check

The `llama-cpp-python` Python wrapper hardcodes supported operating systems and explicitly excludes `sys.platform == 'android'`. 
To bypass this, you need to manually patch `_ctypes_extensions.py` in your site-packages:

1. Locate your site-packages directory (e.g., `/data/data/com.termux/files/usr/lib/python3.13/site-packages/llama_cpp/`)
2. Edit `_ctypes_extensions.py`
3. Modify the platform check to include `android`:
   ```python
   # Original:
   if sys.platform.startswith("linux") or sys.platform.startswith("freebsd"):
   
   # Modified:
   if sys.platform.startswith("linux") or sys.platform.startswith("freebsd") or sys.platform == "android":
   ```

## 5. Missing Codec Dependencies
If the `NeuCodec` dependencies fail to auto-resolve properly, you may need to manually install the specific packages for decoding:

```bash
pip install torchao vector-quantize-pytorch einx local-attention
```

## 6. `soxr` Compilation and Nanobind Stubs

Building `soxr` from source on Android fails due to the `nanobind` stub generator throwing a `PyExc_ImportError` missing symbol exception, as well as a failure to link `libpython` explicitly. To install it:
1. Download and extract the `soxr` source:
   ```bash
   pip download soxr --no-binary :all:
   tar -xzf soxr-*.tar.gz
   cd soxr-*/
   ```
2. Edit `CMakeLists.txt` to remove the `nanobind_add_stub` block.
3. Explicitly link against Python. Change `nanobind_add_module(soxr_ext ...)` to:
   ```cmake
   nanobind_add_module(soxr_ext FREE_THREADED NB_STATIC
       src/soxr_ext.cpp
       ${CSOXR_VER_C}
   )
   target_link_libraries(soxr_ext PRIVATE python3.13)
   ```
4. Install without build isolation:
   ```bash
   pip install . --no-build-isolation --force-reinstall --no-deps
   ```

## 7. `llvmlite` and LLVM Versions

The standard `llvmlite` package on PyPI officially supports up to LLVM 20. Termux ships with LLVM 21, which introduced major breaking changes to the C++ API (e.g., in `orcjit`). Compiling it from source will fail with various C++ errors.
Fortunately, the Termux maintainers provide a heavily patched, pre-compiled package. Bypass `pip` entirely and use `pkg`:

```bash
pkg install python-llvmlite
```

## 8. Bypassing `librosa` (Numba Circular Import Error)

The `numba` library (a dependency of `librosa`) often fails to initialize correctly on Termux, causing a `partially initialized module 'numba' has no attribute 'core'` error during inference.
Since `librosa` is only used to load audio files in `neutts.py`, you can completely bypass it by patching `neutts/neutts.py` to use `torchaudio` instead:

1. Open `neutts/neutts.py`.
2. Remove the `import librosa` statement.
3. Replace the `encode_reference` method with:
   ```python
    def encode_reference(self, ref_audio_path: str | Path):
        import torchaudio
        import torchaudio.functional as F
        wav_tensor, sr = torchaudio.load(ref_audio_path)
        if wav_tensor.shape[0] > 1:
            wav_tensor = wav_tensor.mean(dim=0, keepdim=True)
        if sr != 16000:
            wav_tensor = F.resample(wav_tensor, sr, 16000)
        wav_tensor = wav_tensor.unsqueeze(0)  # [1, 1, T]
        with torch.no_grad():
            ref_codes = self.codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0)
        return ref_codes
   ```

After implementing these fixes, the NeuTTS GGUF backbones should run successfully on Android via Termux!