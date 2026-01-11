import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="[%(asctime)s]: %(message)s:")

list_of_files = [
    "app/__init__.py",
    "app/app.py",
    "app/static/css/style.css",
    "app/templates/index.html",
    "main.py",
    "processors/__init__.py",
    "processors/bert_processing.py",
    "processors/llm_processing.py",
    "processors/regex_processing.py",
    "pyproject.toml",
    "README.md",
    "tests/test_app.py",
    "tests/test_bert.py",
    "tests/test_llm.py",
    "utils/__init__.py",
    "utils/logger.py",
]

for filepath in list_of_files:
    filepath = Path(filepath)
    filedir, filename = os.path.split(filepath)

    if filedir != "":
        os.makedirs(filedir, exist_ok=True)
        logging.info(f"Creating directory: {filedir} for the file {filename}")

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
            logging.info(f"Creating empty file: {filepath}")
    else:
        logging.info(f"File '{filepath}' already exists")
