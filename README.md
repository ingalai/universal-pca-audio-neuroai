# Universal PCA Audio-in-Image NeuroAI Project

This repository contains a Google Colab project that demonstrates audio-in-image encoding and decoding using a universal PCA basis.

## Files

```text
Universal_PCA_EncoderDecoder_GitHub_Colab.ipynb
universal_pca_audio_core_v12.py
universal_pca_basis_current_training_v12.npz
README.md
```

Optional demo files:

```text
demo_cover.png
demo_audio.wav
demo_encoded.png
```

## How to run

1. Open `Universal_PCA_EncoderDecoder_GitHub_Colab.ipynb` in Google Colab.
2. Edit:

```python
REPO_URL = "https://github.com/YOUR_USERNAME/YOUR_REPO_NAME.git"
```

3. Choose:

```python
RUN_MODE = "encode_decode"
```

4. Run all cells.

If demo files are not included, Colab will ask you to upload a cover image and WAV audio.

## Modes

```python
RUN_MODE = "encode"
RUN_MODE = "decode"
RUN_MODE = "encode_decode"
RUN_MODE = "demo"
```

## Important

The encoder and decoder must use the same:

- universal PCA basis `.npz`
- CONFIG
- K
- N payload components
- gamma
- fixed audio sample rate and length

PNG/BMP/TIFF are recommended. JPEG is lossy and may damage the hidden audio signal.