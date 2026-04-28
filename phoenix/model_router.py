import os
import time
import logging
from pathlib import Path
from typing import Dict, Optional, List
from langchain_community.llms.llamacpp import LlamaCpp

logger = logging.getLogger("phoenix.model_router")

class ModelInfo:
    def __init__(self, name: str, path: str, tags: List[str], n_ctx: int, chat_format: Optional[str] = None):
        self.name = name
        self.path = path
        self.tags = tags          # e.g. ["code"], ["reasoning"], ["simple"]
        self.n_ctx = n_ctx
        self.chat_format = chat_format

class ModelRouter:
    def __init__(self, registry_path: Path):
        self.registry_path = registry_path
        self.models: Dict[str, ModelInfo] = {}
        self.loaded_model: Optional[LlamaCpp] = None
        self.loaded_name: Optional[str] = None
        self._load_registry()

    def _load_registry(self):
        """Load model definitions from a JSON file."""
        import json
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
            for entry in data:
                info = ModelInfo(
                    name=entry["name"],
                    path=entry["path"],
                    tags=entry.get("tags", []),
                    n_ctx=entry.get("n_ctx", 2048),
                    chat_format=entry.get("chat_format")
                )
                self.models[entry["name"]] = info
            logger.info(f"Loaded {len(self.models)} model(s) from registry.")

    def save_registry(self):
        """Save current registry to disk."""
        import json
        data = []
        for info in self.models.values():
            data.append({
                "name": info.name,
                "path": info.path,
                "tags": info.tags,
                "n_ctx": info.n_ctx,
                "chat_format": info.chat_format
            })
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def add_model(self, info: ModelInfo):
        self.models[info.name] = info
        self.save_registry()

    def get_model(self, task_hint: str = "reasoning") -> LlamaCpp:
        """Return a loaded model suitable for the given task hint.
        Uses simple tag matching: if task_hint matches any tag, use that model.
        Falls back to a default model.
        """
        # Find matching model by tag
        selected_name = None
        for name, info in self.models.items():
            if task_hint in info.tags:
                selected_name = name
                break
        if selected_name is None:
            # Fallback: pick the first model with "reasoning" tag, else any
            for name, info in self.models.items():
                if "reasoning" in info.tags:
                    selected_name = name
                    break
            if selected_name is None and self.models:
                selected_name = next(iter(self.models))

        if selected_name is None:
            raise ValueError("No models registered. Add models to the registry.")

        # If already loaded the same model, return it
        if self.loaded_name == selected_name and self.loaded_model is not None:
            return self.loaded_model

        # Otherwise, unload current model and load the new one
        self._unload()
        info = self.models[selected_name]
        logger.info(f"Loading model '{info.name}' from {info.path}")
        self.loaded_model = LlamaCpp(
            model_path=info.path,
            temperature=0.7,
            max_tokens=512,           # will be overridden per request if needed
            n_ctx=info.n_ctx,
            chat_format=info.chat_format,
            verbose=False,
            n_gpu_layers=0,           # CPU only for Codespace; set to -1 later for GPU
        )
        self.loaded_name = selected_name
        return self.loaded_model

    def _unload(self):
        """Unload the current model to free memory."""
        if self.loaded_model is not None:
            del self.loaded_model
            self.loaded_model = None
            self.loaded_name = None
            import gc
            gc.collect()

    def list_models(self):
        return [{"name": name, "tags": info.tags} for name, info in self.models.items()]
