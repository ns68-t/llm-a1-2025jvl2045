"""Fetch a submitted adapter from the public Hugging Face archive on demand."""

import os
from pathlib import Path
from zipfile import ZipFile


ADAPTER_REPO = "Neeraj77/llm-a1-adapters"
ARCHIVE_NAME = "All Adapters.zip"
ADAPTERS = {"A1", "A2", "A3", "A4", "A5", "A6", "victim_raw", "victim_san", "best_sanitizer"}


def ensure_adapter(adapter_path):
    """Return a local adapter directory, downloading it if absent."""
    target = Path(adapter_path)
    if (target / "adapter_config.json").is_file() and (target / "adapter_model.safetensors").is_file():
        return target

    name = target.name
    if name not in ADAPTERS:
        raise FileNotFoundError(f"Adapter not found: {target}")

    from huggingface_hub import hf_hub_download

    repo = os.environ.get("LLM_A1_HF_REPO", ADAPTER_REPO)
    archive = hf_hub_download(repo_id=repo, filename=ARCHIVE_NAME, repo_type="model")
    source_name = "A4" if name == "best_sanitizer" else name
    prefix = f"All Adapters/A1 to 6/{source_name}/" if source_name.startswith("A") else f"All Adapters/{source_name}/"
    target.mkdir(parents=True, exist_ok=True)
    with ZipFile(archive) as bundle:
        members = [entry for entry in bundle.infolist() if entry.filename.startswith(prefix) and not entry.is_dir()]
        if not members:
            raise FileNotFoundError(f"{source_name} is missing from {repo}/{ARCHIVE_NAME}")
        for entry in members:
            filename = entry.filename[len(prefix):]
            if "/" in filename or "\\" in filename or filename in ("", ".", ".."):
                raise ValueError(f"Unsafe adapter archive entry: {entry.filename}")
            destination = target / filename
            with bundle.open(entry) as source, destination.open("wb") as output:
                while chunk := source.read(1024 * 1024):
                    output.write(chunk)
    if not (target / "adapter_config.json").is_file() or not (target / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"Incomplete adapter in {repo}/{ARCHIVE_NAME}: {source_name}")
    return target
