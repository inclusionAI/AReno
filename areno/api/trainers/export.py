"""Optional export after successful training, while live weights are available."""

from __future__ import annotations


def export_gguf_if_configured(trainer) -> None:
    path = getattr(trainer.config, "export_gguf_path", None)
    if path is None:
        return
    file_type = trainer.config.gguf_file_type
    trainer.logger.info("stage=export_gguf_start path=%s file_type=%s", path, file_type)
    saved_path = trainer.areno.export_gguf(path, file_type=file_type)
    trainer.logger.info("stage=export_gguf_end path=%s", saved_path)
