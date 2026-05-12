import io
import json
from pathlib import Path

import numpy as np
from PIL import Image

from universal_pca_audio_core import (
    clone_config,
    compare_audio_to_reference,
    compare_image_to_reference,
    crop_to_multiple,
    decode_payload_from_ycc,
    encode_payload_into_ycc,
    get_payload_indices,
    image_to_patches,
    load_basis_npz,
    patches_to_image,
    pil_rgb_to_ycbcr_array,
    project_channel_to_basis,
    read_rgb_image,
    reconstruct_channel_from_scores,
    resample_audio_to_fixed_length,
    validate_basis,
    ycbcr_array_to_rgb_pil,
)


LOSSLESS_IMAGE_FORMATS = ("PNG", "BMP", "TIFF")
LOSSY_IMAGE_FORMATS = ("JPEG",)


def _json_scalar(value):
    if isinstance(value, np.ndarray):
        return value.item()
    return value


def read_basis_metadata(path):
    data = np.load(path, allow_pickle=False)
    config_json = _json_scalar(data["config_json"]) if "config_json" in data else None
    summary_json = _json_scalar(data["summary_json"]) if "summary_json" in data else None
    config = clone_config(json.loads(config_json)) if config_json else clone_config()
    summary = json.loads(summary_json) if summary_json else {}
    return config, summary


def load_basis_bundle(path):
    config, summary = read_basis_metadata(path)
    basis = load_basis_npz(path, config)
    validate_basis(config, basis)
    return {
        "path": str(path),
        "config": config,
        "basis": basis,
        "summary": summary,
    }


def normalize_config(config):
    normalized = clone_config(config)
    patch_dim = int(normalized["patch_size"]) ** 2
    payload_end = int(normalized["K_VISIBLE_FIXED"]) + int(normalized["N_PAYLOAD_COMPONENTS"])
    if payload_end > patch_dim:
        raise ValueError("K_VISIBLE_FIXED + N_PAYLOAD_COMPONENTS must stay within patch_size ** 2.")
    if float(normalized["fixed_audio_seconds"]) <= 0:
        raise ValueError("fixed_audio_seconds must be positive.")
    if int(normalized["fixed_audio_sr"]) <= 0:
        raise ValueError("fixed_audio_sr must be positive.")
    if normalized["payload_strategy"] not in {"repeat_average", "sequential"}:
        raise ValueError("payload_strategy must be 'repeat_average' or 'sequential'.")
    if normalized["embed_mode"] not in {"replace", "additive"}:
        raise ValueError("embed_mode must be 'replace' or 'additive'.")
    get_payload_indices(normalized)
    return normalized


def load_config_json(path):
    return normalize_config(json.loads(Path(path).read_text(encoding="utf-8")))


def config_json_text(config):
    return json.dumps(normalize_config(config), indent=2, ensure_ascii=False)


def prepare_cover_rgb(image):
    if isinstance(image, (str, Path)):
        pil = read_rgb_image(image)
    elif isinstance(image, Image.Image):
        pil = image.convert("RGB")
    else:
        pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
    return pil


def crop_cover_for_config(image, config):
    pil = prepare_cover_rgb(image)
    cropped = crop_to_multiple(np.asarray(pil, dtype=np.uint8), int(config["patch_size"]))
    return cropped


def reconstruct_visible_basis_preview(image, config, basis, visible_components=None):
    config = normalize_config(config)
    validate_basis(config, basis)
    rgb = crop_cover_for_config(image, config)
    ycc = pil_rgb_to_ycbcr_array(Image.fromarray(rgb, mode="RGB"))
    patch_size = int(config["patch_size"])
    keep = int(config["K_VISIBLE_FIXED"] if visible_components is None else visible_components)
    keep = max(1, min(keep, patch_size ** 2))

    preview_ycc = ycc.copy()
    for ci, name in enumerate(config["channel_names"]):
        mean_patch = basis[name]["mean_patch"]
        eigvecs = basis[name]["eigvecs"]
        scores = project_channel_to_basis(ycc[:, :, ci], mean_patch, eigvecs, patch_size)
        scores[:, keep:] = 0.0
        preview_ycc[:, :, ci] = reconstruct_channel_from_scores(
            scores,
            mean_patch,
            eigvecs,
            ycc[:, :, ci].shape,
            patch_size,
        )

    preview_pil = ycbcr_array_to_rgb_pil(preview_ycc)
    preview_rgb = np.asarray(preview_pil, dtype=np.uint8)
    return {
        "cover_rgb": rgb,
        "preview_rgb": preview_rgb,
        "metrics": compare_image_to_reference(rgb, preview_rgb),
    }


def encode_audio_workflow(image, audio_samples, audio_sr, config, basis):
    config = normalize_config(config)
    validate_basis(config, basis)
    cover_rgb = crop_cover_for_config(image, config)
    cover_pil = Image.fromarray(cover_rgb, mode="RGB")
    cover_ycc = pil_rgb_to_ycbcr_array(cover_pil)
    audio_fixed = resample_audio_to_fixed_length(
        audio_samples,
        original_sr=int(audio_sr),
        fixed_sr=int(config["fixed_audio_sr"]),
        fixed_len=int(config["fixed_audio_len"]),
    )

    encoded_ycc = encode_payload_into_ycc(cover_ycc, audio_fixed, config, basis)
    encoded_pil = ycbcr_array_to_rgb_pil(encoded_ycc)
    encoded_rgb = np.asarray(encoded_pil, dtype=np.uint8)

    internal_audio = decode_payload_from_ycc(encoded_ycc, config, basis, apply_postprocess=False)
    roundtrip_ycc = pil_rgb_to_ycbcr_array(encoded_pil)
    roundtrip_audio = decode_payload_from_ycc(roundtrip_ycc, config, basis, apply_postprocess=True)

    return {
        "config": config,
        "cover_rgb": cover_rgb,
        "cover_ycc": cover_ycc,
        "audio_fixed": audio_fixed,
        "encoded_ycc": encoded_ycc,
        "encoded_pil": encoded_pil,
        "encoded_rgb": encoded_rgb,
        "internal_audio": internal_audio,
        "roundtrip_audio": roundtrip_audio,
        "image_metrics": compare_image_to_reference(cover_rgb, encoded_rgb),
        "internal_audio_metrics": compare_audio_to_reference(audio_fixed, internal_audio),
        "roundtrip_audio_metrics": compare_audio_to_reference(audio_fixed, roundtrip_audio),
    }


def _roundtrip_rgb_through_format(encoded_rgb, image_format):
    pil = Image.fromarray(np.asarray(encoded_rgb, dtype=np.uint8), mode="RGB")
    buffer = io.BytesIO()
    save_kwargs = {"quality": 95} if image_format == "JPEG" else {}
    pil.save(buffer, format=image_format, **save_kwargs)
    buffer.seek(0)
    return np.asarray(Image.open(buffer).convert("RGB"), dtype=np.uint8), buffer.getvalue()


def decode_image_workflow(encoded_image, config, basis):
    config = normalize_config(config)
    validate_basis(config, basis)
    encoded_rgb = crop_cover_for_config(encoded_image, config)
    encoded_ycc = pil_rgb_to_ycbcr_array(Image.fromarray(encoded_rgb, mode="RGB"))
    decoded_audio = decode_payload_from_ycc(encoded_ycc, config, basis, apply_postprocess=True)
    return {
        "encoded_rgb": encoded_rgb,
        "decoded_audio": decoded_audio,
    }


def compare_decoder_compression_formats(encoded_image, config, basis):
    baseline = decode_image_workflow(encoded_image, config, basis)
    encoded_rgb = baseline["encoded_rgb"]
    decoded_reference = baseline["decoded_audio"]
    comparisons = {}

    for image_format in LOSSLESS_IMAGE_FORMATS + LOSSY_IMAGE_FORMATS:
        roundtrip_rgb, encoded_bytes = _roundtrip_rgb_through_format(encoded_rgb, image_format)
        decoded = decode_image_workflow(roundtrip_rgb, config, basis)["decoded_audio"]
        comparisons[image_format] = {
            "decoded_audio": decoded,
            "metrics_vs_baseline": compare_audio_to_reference(decoded_reference, decoded),
            "image_metrics_vs_input": compare_image_to_reference(encoded_rgb, roundtrip_rgb),
            "encoded_size_bytes": len(encoded_bytes),
        }

    return {
        "baseline": baseline,
        "formats": comparisons,
    }


def basis_tile_grid(basis, channel_name, count=16):
    eigvecs = basis[channel_name]["eigvecs"]
    patch_dim = eigvecs.shape[0]
    patch_size = int(np.sqrt(patch_dim))
    count = max(1, min(int(count), eigvecs.shape[1]))
    tiles = []
    for idx in range(count):
        tile = eigvecs[:, idx].reshape(patch_size, patch_size)
        lo = float(tile.min())
        hi = float(tile.max())
        tile = np.zeros_like(tile) if abs(hi - lo) < 1e-12 else (tile - lo) / (hi - lo)
        tiles.append(tile)
    return np.stack(tiles, axis=0)


def absdiff_image(a, b, scale=8.0):
    diff = np.abs(np.asarray(a, dtype=np.float32) - np.asarray(b, dtype=np.float32)) * float(scale)
    return np.clip(diff, 0, 255).astype(np.uint8)
