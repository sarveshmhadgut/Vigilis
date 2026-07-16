import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

list_of_files = [
    "app/__init__.py",
    "app/app.py",
    "app/static/css/style.css",
    "app/templates/index.html",
    "main.py",
    "processors/__init__.py",
    "processors/bert_processor.py",
    "processors/llm_processor.py",
    "processors/regex_processor.py",
    "pyproject.toml",
    "README.md",
    "tests/test_app.py",
    "tests/test_bert.py",
    "tests/test_llm.py",
    "utils/__init__.py",
    "utils/logger.py",
]

for filepath_str in list_of_files:
    filepath = Path(filepath_str)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logger.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        Path(filepath).touch()
        logger.info(f"Creating empty file: {filepath}")
    else:
        logger.info(f"File '{filepath}' already exists")
