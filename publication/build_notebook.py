"""Convert the readable percent-format source to a standalone GitHub notebook."""
import ast
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "MAGSNet-Nested-5x5-CV.ipynb"


def build():
    cells = []
    for chunk in (HERE / "magsnet_nested_cv.py").read_text(encoding="utf-8").split("# %%"):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            source = "\n".join(line[2:] if line.startswith("# ") else line[1:] if line == "#" else line
                               for line in chunk.splitlines()[1:]).strip()
            cells.append(nbformat.v4.new_markdown_cell(source))
        else:
            source = chunk.strip()
            ast.parse(source)
            cells.append(nbformat.v4.new_code_cell(source))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (dl)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.18"},
    })
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Built {OUTPUT.name}: {len(cells)} cells; all training switches are False.")
    return OUTPUT


if __name__ == "__main__":
    build()
