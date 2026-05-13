import io
import json
import wave
from pathlib import Path

import numpy as np
from PIL import Image


DEFAULT_CONFIG = {
    "patch_size": 8,
    "channel_names": ["Y", "Cb", "Cr"],
    "embed_channels": ["Cb", "Cr"],
    "fixed_audio_sr": 4000,
    "fixed_audio_seconds": 5.0,
    "max_patches_per_channel": None,
    "K_VISIBLE_FIXED": 16,
    "N_PAYLOAD_COMPONENTS": 16,
    "gamma": {
        "Cb": 80.0,
        "Cr": 80.0,
    },
    "embed_mode": "replace",
    "payload_strategy": "repeat_average",
    "payload_repeat_lanes": 8,
    "decoded_postprocess": {
        "enabled": False,
        "remove_dc": True,
        "lowpass_hz": 1600.0,
        "fir_taps": 41,
        "renormalize_peak": False,
    },
}


def clone_config(config=None):
    base = DEFAULT_CONFIG if config is None else config
    copied = json.loads(json.dumps(base))
    copied["fixed_audio_len"] = int(copied["fixed_audio_sr"] * copied["fixed_audio_seconds"])
    return copied


def read_rgb_image_from_bytes(data):
    return Image.open(io.BytesIO(data)).convert("RGB")


def read_rgb_image(path):
    return Image.open(path).convert("RGB")


def crop_to_multiple(arr, patch_size):
    h, w = arr.shape[:2]
    h2 = (h // patch_size) * patch_size
    w2 = (w // patch_size) * patch_size
    if h2 <= 0 or w2 <= 0:
        raise ValueError(f"Image is too small for patch_size={patch_size}.")
    return arr[:h2, :w2, ...]


def pil_rgb_to_ycbcr_array(img_pil):
    return np.asarray(img_pil.convert("YCbCr"), dtype=np.float32)


def ycbcr_array_to_rgb_pil(ycbcr_arr):
    arr = np.clip(ycbcr_arr, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, mode="YCbCr").convert("RGB")


def read_wav_mono_from_bytes(data):
    with wave.open(io.BytesIO(data), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    return pcm_bytes_to_float32_mono(raw, n_channels, sampwidth), sample_rate


def read_wav_mono(path):
    with wave.open(str(path), "rb") as wf:
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        sample_rate = wf.getframerate()
        n_frames = wf.getnframes()
        raw = wf.readframes(n_frames)
    return pcm_bytes_to_float32_mono(raw, n_channels, sampwidth), sample_rate


def pcm_bytes_to_float32_mono(raw, n_channels, sampwidth):
    if sampwidth == 1:
        samples = np.frombuffer(raw, dtype=np.uint8).astype(np.float32)
        samples = (samples - 128.0) / 128.0
    elif sampwidth == 2:
        samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 4:
        samples = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ValueError("Only 8-bit, 16-bit, or 32-bit PCM WAV is supported.")
    if n_channels > 1:
        samples = samples.reshape(-1, n_channels).mean(axis=1)
    return samples.astype(np.float32)


def write_wav_bytes_from_float32(samples, sample_rate):
    samples = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    int16_samples = (samples * 32767.0).astype(np.int16)
    bio = io.BytesIO()
    with wave.open(bio, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(int(sample_rate))
        wf.writeframes(int16_samples.tobytes())
    bio.seek(0)
    return bio.getvalue()


def write_wav_float32(path, samples, sample_rate):
    Path(path).write_bytes(write_wav_bytes_from_float32(samples, sample_rate))


def normalize_audio(x):
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    peak = float(np.max(np.abs(x)))
    return x if peak < 1e-9 else (x / peak).astype(np.float32)


def resample_audio_to_fixed_length(audio, original_sr, fixed_sr, fixed_len):
    audio = normalize_audio(audio)
    if len(audio) == 0:
        return np.zeros(fixed_len, dtype=np.float32)
    duration = len(audio) / float(original_sr)
    target_len = max(1, int(round(duration * fixed_sr)))
    x_old = np.linspace(0.0, duration, num=len(audio), endpoint=False)
    x_new = np.linspace(0.0, duration, num=target_len, endpoint=False)
    resampled = np.interp(x_new, x_old, audio).astype(np.float32)
    out = np.zeros(fixed_len, dtype=np.float32)
    n = min(fixed_len, len(resampled))
    out[:n] = resampled[:n]
    return normalize_audio(out)


def image_to_patches(img, patch_size):
    img = np.asarray(img, dtype=np.float32)
    h, w = img.shape
    if h % patch_size != 0 or w % patch_size != 0:
        raise ValueError("Image channel shape must be a multiple of patch_size.")
    return (
        img.reshape(h // patch_size, patch_size, w // patch_size, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(-1, patch_size * patch_size)
        .astype(np.float32)
    )


def patches_to_image(patches, shape, patch_size):
    h, w = shape
    return (
        np.asarray(patches, dtype=np.float32)
        .reshape(h // patch_size, w // patch_size, patch_size, patch_size)
        .transpose(0, 2, 1, 3)
        .reshape(h, w)
        .astype(np.float32)
    )


def fit_pca_from_patches(X):
    X = np.asarray(X, dtype=np.float32)
    mean_patch = X.mean(axis=0)
    Xc = X - mean_patch
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)
    total = float(np.sum(eigvals))
    explained = eigvals / total if total > 1e-12 else np.zeros_like(eigvals)
    return mean_patch.astype(np.float32), eigvecs, eigvals, explained.astype(np.float32)


def pca_from_mean_and_cov(mean_patch, cov, count):
    cov = np.asarray(cov, dtype=np.float64) / max(1, int(count) - 1)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order].astype(np.float32)
    eigvecs = eigvecs[:, order].astype(np.float32)
    total = float(np.sum(eigvals))
    explained = eigvals / total if total > 1e-12 else np.zeros_like(eigvals)
    return mean_patch.astype(np.float32), eigvecs, eigvals, explained.astype(np.float32)



def fit_pca_oja_from_patches(
    X,
    n_components=None,
    learning_rate=0.01,
    epochs=5,
    batch_size=512,
    seed=0,
    shuffle=True,
    orthonormalize_every=1,
):
    """Train PCA-like basis with Hebbian learning + Oja / Sanger GHA.

    This implements the lecture-style idea:
        y = W^T x
        Δw_i = η y_i (x - Σ_{j<=i} y_j w_j)

    The second term is the Oja/Sanger normalization and lateral-inhibition term.
    It prevents weights from growing without bound and encourages different
    neurons/columns to learn different principal components.

    For numerical stability in a notebook, this implementation uses mini-batches
    and periodically orthonormalizes the weight matrix with QR decomposition.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.ndim != 2:
        raise ValueError("X must be a 2D matrix: [num_patches, patch_dim].")
    if X.shape[0] < 2:
        raise ValueError("Need at least two patches to train PCA basis.")

    mean_patch = X.mean(axis=0).astype(np.float32)
    Xc = (X - mean_patch).astype(np.float32)
    num_samples, dim = Xc.shape
    k = dim if n_components is None else int(n_components)
    if k <= 0 or k > dim:
        raise ValueError("n_components must be between 1 and patch_dim.")

    rng = np.random.default_rng(seed)
    W = rng.normal(0.0, 1.0, size=(dim, k)).astype(np.float32)
    W, _ = np.linalg.qr(W)
    W = W[:, :k].astype(np.float32)

    lr0 = float(learning_rate)
    batch_size = max(1, int(batch_size))
    epochs = max(1, int(epochs))

    for epoch in range(epochs):
        if shuffle:
            order = rng.permutation(num_samples)
        else:
            order = np.arange(num_samples)

        # A mild decay makes late updates more stable.
        lr = lr0 / np.sqrt(epoch + 1.0)

        for start in range(0, num_samples, batch_size):
            idx = order[start:start + batch_size]
            Xb = Xc[idx].astype(np.float32)
            Y = Xb @ W  # [batch, k]

            # Sanger / Generalized Hebbian Algorithm.
            # For each component i, subtract projections onto components 0..i.
            dW = np.zeros_like(W, dtype=np.float32)
            for i in range(k):
                yi = Y[:, i:i + 1]  # [batch, 1]
                reconstruction_i = Y[:, :i + 1] @ W[:, :i + 1].T
                residual_i = Xb - reconstruction_i
                dW[:, i] = (yi * residual_i).mean(axis=0)

            W = W + lr * dW

        if orthonormalize_every and ((epoch + 1) % int(orthonormalize_every) == 0):
            W, _ = np.linalg.qr(W)
            W = W[:, :k].astype(np.float32)

    # Complete to a full dim x dim basis if only k components were trained.
    if k < dim:
        Q, _ = np.linalg.qr(np.concatenate([W, np.eye(dim, dtype=np.float32)], axis=1))
        eigvecs = Q[:, :dim].astype(np.float32)
    else:
        eigvecs = W.astype(np.float32)

    # Estimate eigenvalues/explained variance by projecting data onto learned axes.
    scores = Xc @ eigvecs
    eigvals = np.var(scores, axis=0, ddof=1).astype(np.float32)
    order = np.argsort(eigvals)[::-1]
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order].astype(np.float32)
    total = float(np.sum(eigvals))
    explained = eigvals / total if total > 1e-12 else np.zeros_like(eigvals)
    return mean_patch, eigvecs, eigvals, explained.astype(np.float32)


def train_universal_pca_oja(training_dir, config=None, seed=None, progress_callback=None):
    """Train universal PCA basis using lecture-style Hebbian/Oja learning.

    This function is intentionally different from the closed-form eigen-decomposition
    trainer. It collects image patches, then learns the PCA axes iteratively using
    Oja/Sanger's rule.
    """
    config = clone_config(config)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in Path(training_dir).iterdir() if p.suffix.lower() in image_exts)
    if not paths:
        raise RuntimeError(f"No training images found in {training_dir}.")

    rng = np.random.default_rng(seed) if seed is not None else np.random.default_rng()
    max_patches = config.get("max_patches_per_channel", 50000)
    if max_patches in (None, 0, "all", "ALL"):
        max_patches = None
    else:
        max_patches = int(max_patches)

    per_image_patch_cap = None if max_patches is None else max(1, int(np.ceil(max_patches / len(paths))))
    patch_bank = {name: [] for name in config["channel_names"]}

    def report(message, done=0, total=1):
        if progress_callback is not None:
            progress_callback(message, done, total)

    for image_index, path in enumerate(paths, start=1):
        report(f"Oja patch collection {image_index}/{len(paths)}: {path.name}", image_index, len(paths) + len(config["channel_names"]))
        rgb = crop_to_multiple(np.asarray(read_rgb_image(path), dtype=np.uint8), config["patch_size"])
        ycc = pil_rgb_to_ycbcr_array(Image.fromarray(rgb, mode="RGB"))
        for ci, name in enumerate(config["channel_names"]):
            patches = image_to_patches(ycc[:, :, ci], config["patch_size"])
            if per_image_patch_cap is not None and len(patches) > per_image_patch_cap:
                idx = rng.choice(len(patches), size=per_image_patch_cap, replace=False)
                patches = patches[idx]
            patch_bank[name].append(patches.astype(np.float32))

    basis = {}
    summary = {}
    last_payload_pc = config["K_VISIBLE_FIXED"] + config["N_PAYLOAD_COMPONENTS"] - 1
    patch_dim = int(config["patch_size"]) ** 2
    n_components = int(config.get("oja_components", patch_dim))
    n_components = min(max(1, n_components), patch_dim)

    for channel_index, name in enumerate(config["channel_names"], start=1):
        X = np.vstack(patch_bank[name]).astype(np.float32)
        if max_patches is not None and len(X) > max_patches:
            idx = rng.choice(len(X), size=max_patches, replace=False)
            X = X[idx]

        report(f"Oja training channel {name} with {len(X)} patches", len(paths) + channel_index, len(paths) + len(config["channel_names"]))
        mean_patch, eigvecs, eigvals, explained = fit_pca_oja_from_patches(
            X,
            n_components=n_components,
            learning_rate=float(config.get("oja_learning_rate", 0.01)),
            epochs=int(config.get("oja_epochs", 5)),
            batch_size=int(config.get("oja_batch_size", 512)),
            seed=None if seed is None else int(seed) + channel_index,
            orthonormalize_every=int(config.get("oja_orthonormalize_every", 1)),
        )
        basis[name] = {
            "mean_patch": mean_patch,
            "eigvecs": eigvecs,
            "eigvals": eigvals,
            "explained_variance": explained,
        }
        summary[name] = {
            "patches": list(X.shape),
            "cum_explained_to_last_payload_pc": float(np.cumsum(explained)[last_payload_pc]),
            "training_mode": "hebbian_oja_sanger_rule",
            "oja_epochs": int(config.get("oja_epochs", 5)),
            "oja_learning_rate": float(config.get("oja_learning_rate", 0.01)),
            "oja_batch_size": int(config.get("oja_batch_size", 512)),
            "oja_components": int(n_components),
        }

    report("Oja PCA basis fitted", 1, 1)
    return basis, summary

def project_channel_to_basis(img, mean_patch, eigvecs, patch_size):
    X = image_to_patches(img, patch_size)
    return (X - mean_patch) @ eigvecs


def reconstruct_channel_from_scores(scores, mean_patch, eigvecs, shape, patch_size):
    X_rec = np.asarray(scores, dtype=np.float32) @ eigvecs.T + mean_patch
    return patches_to_image(X_rec, shape, patch_size)


def get_payload_indices(config):
    K = int(config["K_VISIBLE_FIXED"])
    N = int(config["N_PAYLOAD_COMPONENTS"])
    if K < 0 or N <= 0:
        raise ValueError("K_VISIBLE_FIXED must be >= 0 and N_PAYLOAD_COMPONENTS must be > 0.")
    if K + N > int(config["patch_size"]) ** 2:
        raise ValueError("K_VISIBLE_FIXED + N_PAYLOAD_COMPONENTS must be <= patch_size ** 2.")
    return list(range(K, K + N))


def get_payload_lanes(config):
    lanes = []
    for ch_name in config["embed_channels"]:
        for pc_index in get_payload_indices(config):
            lanes.append((ch_name, pc_index))
    if config.get("payload_strategy", "sequential") == "repeat_average":
        repeat_lanes = int(config.get("payload_repeat_lanes", len(lanes)))
        if repeat_lanes <= 0:
            raise ValueError("payload_repeat_lanes must be positive.")
        return lanes[: min(repeat_lanes, len(lanes))]
    return lanes


def get_audio_patch_positions(num_patches, fixed_audio_len):
    if fixed_audio_len > num_patches:
        raise ValueError(
            "repeat_average requires fixed_audio_len <= number of patches per image. "
            "Use a larger cover image, shorter audio, or sequential payload_strategy."
        )
    if fixed_audio_len == num_patches:
        return np.arange(num_patches, dtype=np.int64)
    return np.linspace(0, num_patches - 1, fixed_audio_len).round().astype(np.int64)


def compute_capacity_for_image_shape(h, w, config):
    patch_size = int(config["patch_size"])
    num_patches = (h // patch_size) * (w // patch_size)
    capacity = num_patches * len(config["embed_channels"]) * len(get_payload_indices(config))
    return int(capacity), int(num_patches)


def validate_basis(config, basis):
    patch_dim = int(config["patch_size"]) ** 2
    for name in config["channel_names"]:
        if name not in basis:
            raise ValueError(f"Missing PCA basis for channel {name}.")
        if basis[name]["mean_patch"].shape[0] != patch_dim:
            raise ValueError(f"mean_patch for {name} has wrong length.")
        if basis[name]["eigvecs"].shape != (patch_dim, patch_dim):
            raise ValueError(f"eigvecs for {name} must have shape {(patch_dim, patch_dim)}.")


def encode_payload_into_ycc(cover_ycc, payload_audio, config, basis):
    validate_basis(config, basis)
    patch_size = config["patch_size"]
    payload_indices = get_payload_indices(config)
    embed_mode = config["embed_mode"]
    if embed_mode not in {"replace", "additive"}:
        raise ValueError("embed_mode must be 'replace' or 'additive'.")

    cover_ycc = np.asarray(cover_ycc, dtype=np.float32)
    h, w, _ = cover_ycc.shape
    capacity, num_patches = compute_capacity_for_image_shape(h, w, config)
    if capacity < config["fixed_audio_len"]:
        raise ValueError(f"Image capacity {capacity} is smaller than fixed_audio_len.")

    audio = np.asarray(payload_audio, dtype=np.float32).ravel()
    audio_fixed = np.zeros(config["fixed_audio_len"], dtype=np.float32)
    audio_fixed[: min(len(audio), config["fixed_audio_len"])] = audio[: config["fixed_audio_len"]]

    out_ycc = cover_ycc.copy()
    strategy = config.get("payload_strategy", "sequential")
    if strategy == "repeat_average":
        patch_positions = get_audio_patch_positions(num_patches, config["fixed_audio_len"])
        lane_set = set(get_payload_lanes(config))
        for ch_name in config["embed_channels"]:
            ci = config["channel_names"].index(ch_name)
            mean_patch = basis[ch_name]["mean_patch"]
            eigvecs = basis[ch_name]["eigvecs"]
            scores = project_channel_to_basis(out_ycc[:, :, ci], mean_patch, eigvecs, patch_size)
            gamma = float(config["gamma"][ch_name])
            for pi in payload_indices:
                if (ch_name, pi) not in lane_set:
                    continue
                if embed_mode == "replace":
                    scores[patch_positions, pi] = gamma * audio_fixed
                else:
                    scores[patch_positions, pi] = scores[patch_positions, pi] + gamma * audio_fixed
            out_ycc[:, :, ci] = reconstruct_channel_from_scores(
                scores, mean_patch, eigvecs, out_ycc[:, :, ci].shape, patch_size
            )
        return out_ycc.astype(np.float32)

    if strategy != "sequential":
        raise ValueError("payload_strategy must be 'sequential' or 'repeat_average'.")

    payload = np.zeros(capacity, dtype=np.float32)
    payload[: config["fixed_audio_len"]] = audio_fixed
    payload_offset = 0
    for ch_name in config["embed_channels"]:
        ci = config["channel_names"].index(ch_name)
        mean_patch = basis[ch_name]["mean_patch"]
        eigvecs = basis[ch_name]["eigvecs"]
        scores = project_channel_to_basis(out_ycc[:, :, ci], mean_patch, eigvecs, patch_size)
        gamma = float(config["gamma"][ch_name])
        for pi in payload_indices:
            n = scores.shape[0]
            vals = payload[payload_offset : payload_offset + n]
            if len(vals) < n:
                vals = np.pad(vals, (0, n - len(vals)))
            if embed_mode == "replace":
                scores[:, pi] = gamma * vals
            else:
                scores[:, pi] = scores[:, pi] + gamma * vals
            payload_offset += n
        out_ycc[:, :, ci] = reconstruct_channel_from_scores(
            scores, mean_patch, eigvecs, out_ycc[:, :, ci].shape, patch_size
        )
    return out_ycc.astype(np.float32)


def decode_payload_from_ycc(encoded_ycc, config, basis, apply_postprocess=True):
    validate_basis(config, basis)
    patch_size = config["patch_size"]
    payload_indices = get_payload_indices(config)
    encoded_ycc = np.asarray(encoded_ycc, dtype=np.float32)
    h, w, _ = encoded_ycc.shape
    capacity, num_patches = compute_capacity_for_image_shape(h, w, config)
    if capacity < config["fixed_audio_len"]:
        raise ValueError(f"Image capacity {capacity} is smaller than fixed_audio_len.")

    strategy = config.get("payload_strategy", "sequential")
    if strategy == "repeat_average":
        patch_positions = get_audio_patch_positions(num_patches, config["fixed_audio_len"])
        decoded_lanes = []
        for ch_name, pi in get_payload_lanes(config):
            ci = config["channel_names"].index(ch_name)
            mean_patch = basis[ch_name]["mean_patch"]
            eigvecs = basis[ch_name]["eigvecs"]
            scores = project_channel_to_basis(encoded_ycc[:, :, ci], mean_patch, eigvecs, patch_size)
            gamma = float(config["gamma"][ch_name])
            decoded_lanes.append(scores[patch_positions, pi] / gamma)
        decoded_audio = np.mean(np.vstack(decoded_lanes), axis=0).astype(np.float32)
        if apply_postprocess:
            decoded_audio = postprocess_decoded_audio(decoded_audio, config)
        return np.clip(decoded_audio, -1.0, 1.0).astype(np.float32)

    if strategy != "sequential":
        raise ValueError("payload_strategy must be 'sequential' or 'repeat_average'.")

    decoded_values = []
    for ch_name in config["embed_channels"]:
        ci = config["channel_names"].index(ch_name)
        mean_patch = basis[ch_name]["mean_patch"]
        eigvecs = basis[ch_name]["eigvecs"]
        scores = project_channel_to_basis(encoded_ycc[:, :, ci], mean_patch, eigvecs, patch_size)
        gamma = float(config["gamma"][ch_name])
        for pi in payload_indices:
            decoded_values.append(scores[:, pi] / gamma)
    decoded_audio = np.concatenate(decoded_values).astype(np.float32)[: config["fixed_audio_len"]]
    if apply_postprocess:
        decoded_audio = postprocess_decoded_audio(decoded_audio, config)
    return np.clip(decoded_audio, -1.0, 1.0).astype(np.float32)


def train_universal_pca(training_dir, config=None, seed=None, progress_callback=None):
    config = clone_config(config)
    training_method = str(config.get("pca_training_method", "eigh")).lower()
    if training_method in ("oja", "hebbian", "hebbian_oja", "sanger", "gha"):
        return train_universal_pca_oja(training_dir, config, seed=seed, progress_callback=progress_callback)
    image_exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    paths = sorted(p for p in Path(training_dir).iterdir() if p.suffix.lower() in image_exts)
    if not paths:
        raise RuntimeError(f"No training images found in {training_dir}.")

    patch_bank = {name: [] for name in config["channel_names"]}
    max_patches = config.get("max_patches_per_channel", None)
    use_all_patches = max_patches in (None, 0, "all", "ALL")
    rng = np.random.default_rng(seed if seed is not None else config.get("random_seed", 0))
    per_image_patch_cap = None
    if not use_all_patches:
        per_image_patch_cap = max(1, int(np.ceil(int(max_patches) / len(paths))))

    def report(message, done=0, total=1):
        if progress_callback is not None:
            progress_callback(message, done, total)

    if use_all_patches:
        patch_dim = int(config["patch_size"]) ** 2
        sums = {name: np.zeros(patch_dim, dtype=np.float64) for name in config["channel_names"]}
        counts = {name: 0 for name in config["channel_names"]}
        total_steps = len(paths) * 2

        for image_index, path in enumerate(paths, start=1):
            report(f"mean pass {image_index}/{len(paths)}: {path.name}", image_index, total_steps)
            rgb = crop_to_multiple(np.asarray(read_rgb_image(path), dtype=np.uint8), config["patch_size"])
            ycc = pil_rgb_to_ycbcr_array(Image.fromarray(rgb, mode="RGB"))
            for ci, name in enumerate(config["channel_names"]):
                patches = image_to_patches(ycc[:, :, ci], config["patch_size"]).astype(np.float64)
                sums[name] += patches.sum(axis=0)
                counts[name] += int(patches.shape[0])

        means = {name: sums[name] / max(1, counts[name]) for name in config["channel_names"]}
        covs = {name: np.zeros((patch_dim, patch_dim), dtype=np.float64) for name in config["channel_names"]}

        for image_index, path in enumerate(paths, start=1):
            report(
                f"covariance pass {image_index}/{len(paths)}: {path.name}",
                len(paths) + image_index,
                total_steps,
            )
            rgb = crop_to_multiple(np.asarray(read_rgb_image(path), dtype=np.uint8), config["patch_size"])
            ycc = pil_rgb_to_ycbcr_array(Image.fromarray(rgb, mode="RGB"))
            for ci, name in enumerate(config["channel_names"]):
                patches = image_to_patches(ycc[:, :, ci], config["patch_size"]).astype(np.float64)
                centered = patches - means[name]
                covs[name] += centered.T @ centered

        basis = {}
        summary = {}
        last_payload_pc = config["K_VISIBLE_FIXED"] + config["N_PAYLOAD_COMPONENTS"] - 1
        for name in config["channel_names"]:
            mean_patch, eigvecs, eigvals, explained = pca_from_mean_and_cov(means[name], covs[name], counts[name])
            basis[name] = {
                "mean_patch": mean_patch,
                "eigvecs": eigvecs,
                "eigvals": eigvals,
                "explained_variance": explained,
            }
            summary[name] = {
                "patches": [counts[name], patch_dim],
                "cum_explained_to_last_payload_pc": float(np.cumsum(explained)[last_payload_pc]),
                "training_mode": "all_patches_streaming",
            }
        report("PCA basis fitted", total_steps, total_steps)
        return basis, summary

    for path in paths:
        rgb = crop_to_multiple(np.asarray(read_rgb_image(path), dtype=np.uint8), config["patch_size"])
        ycc = pil_rgb_to_ycbcr_array(Image.fromarray(rgb, mode="RGB"))
        for ci, name in enumerate(config["channel_names"]):
            patches = image_to_patches(ycc[:, :, ci], config["patch_size"])
            if per_image_patch_cap is not None and len(patches) > per_image_patch_cap:
                idx = rng.choice(len(patches), size=per_image_patch_cap, replace=False)
                patches = patches[idx]
            patch_bank[name].append(patches)

    basis = {}
    summary = {}
    last_payload_pc = config["K_VISIBLE_FIXED"] + config["N_PAYLOAD_COMPONENTS"] - 1
    for name in config["channel_names"]:
        X = np.vstack(patch_bank[name]).astype(np.float32)
        if not use_all_patches and len(X) > int(max_patches):
            idx = rng.choice(len(X), size=int(max_patches), replace=False)
            X = X[idx]
        mean_patch, eigvecs, eigvals, explained = fit_pca_from_patches(X)
        basis[name] = {
            "mean_patch": mean_patch,
            "eigvecs": eigvecs,
            "eigvals": eigvals,
            "explained_variance": explained,
        }
        summary[name] = {
            "patches": list(X.shape),
            "cum_explained_to_last_payload_pc": float(np.cumsum(explained)[last_payload_pc]),
        }
    return basis, summary


def save_basis_npz(path, basis, config=None, summary=None):
    config = clone_config(config)
    payload = {"config_json": np.array(json.dumps(config))}
    if summary is not None:
        payload["summary_json"] = np.array(json.dumps(summary))
    for name in config["channel_names"]:
        payload[f"mean_patch_{name}"] = basis[name]["mean_patch"]
        payload[f"eigvecs_{name}"] = basis[name]["eigvecs"]
        payload[f"eigvals_{name}"] = basis[name]["eigvals"]
    np.savez(path, **payload)


def load_basis_npz(path, config=None):
    config = clone_config(config)
    data = np.load(path, allow_pickle=False)
    basis = {}
    for name in config["channel_names"]:
        eigvals = data[f"eigvals_{name}"].astype(np.float32)
        total = float(np.sum(eigvals))
        basis[name] = {
            "mean_patch": data[f"mean_patch_{name}"].astype(np.float32),
            "eigvecs": data[f"eigvecs_{name}"].astype(np.float32),
            "eigvals": eigvals,
            "explained_variance": eigvals / total if total > 1e-12 else np.zeros_like(eigvals),
        }
    validate_basis(config, basis)
    return basis


# Evaluation metric helpers were removed in v4 because the notebook now focuses on the
# qualitative encoding/decoding workflow rather than numeric indicators.

def postprocess_decoded_audio(samples, config):
    settings = config.get("decoded_postprocess", {})
    if not settings.get("enabled", False):
        return np.asarray(samples, dtype=np.float32)
    out = np.asarray(samples, dtype=np.float32).copy()
    if settings.get("remove_dc", True):
        out = out - float(np.mean(out))
    out = lowpass_fir(
        out,
        sample_rate=config["fixed_audio_sr"],
        cutoff_hz=settings.get("lowpass_hz", None),
        taps=settings.get("fir_taps", 41),
    )
    if settings.get("renormalize_peak", False):
        out = normalize_audio(out)
    return np.clip(out, -1.0, 1.0).astype(np.float32)
