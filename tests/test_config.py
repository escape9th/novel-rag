from pathlib import Path

import config


def test_config_defaults():
    assert isinstance(config.DATA_DIR, Path)
    assert config.DATA_DIR.is_absolute()
    assert config.TOP_K >= 1
    assert config.CHUNK_SIZE > config.CHUNK_OVERLAP >= 0
    assert config.DEEPSEEK_MODEL
    assert config.EMBED_MODEL_NAME


def test_api_key_is_string():
    assert isinstance(config.DEEPSEEK_API_KEY, str)
