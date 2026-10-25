"""Build the standalone inference notebook. Existing executed notebooks are protected."""
import ast
from pathlib import Path

import nbformat

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "06_BRISC2025_External_Validation.ipynb"


def build():
    if OUTPUT.exists():
        existing = nbformat.read(OUTPUT, as_version=4)
        if any(c.get("execution_count") is not None or c.get("outputs") for c in existing.cells):
            raise RuntimeError("Refusing to overwrite an executed notebook. Edit its cells directly.")
    cells = []
    for chunk in (HERE / "brisc_external_validation.py").read_text(encoding="utf-8").split("# %%"):
        if not chunk.strip():
            continue
        if chunk.startswith(" [markdown]"):
            text = "\n".join(line[2:] if line.startswith("# ") else line for line in chunk.splitlines()[1:]).strip()
            cells.append(nbformat.v4.new_markdown_cell(text))
        else:
            ast.parse(chunk.strip())
            cells.append(nbformat.v4.new_code_cell(chunk.strip()))
    notebook = nbformat.v4.new_notebook(cells=cells, metadata={
        "kernelspec": {"display_name": "Python (dl)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.10.18"}})
    nbformat.validate(notebook)
    nbformat.write(notebook, OUTPUT)
    print(f"Created {OUTPUT.name}: {len(cells)} cells; inference only.")
    return OUTPUT


if __name__ == "__main__":
    build()
