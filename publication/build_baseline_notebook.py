"""Build the standalone baseline notebook without modifying the MAGS-Net notebook."""
import ast
from pathlib import Path
import nbformat

HERE = Path(__file__).resolve().parent
OUTPUT = HERE.parent / "02_Baseline_Classification_NestedCV.ipynb"


def build():
    cells = []
    for chunk in (HERE / "baseline_classification.py").read_text(encoding="utf-8").split("# %%"):
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
    print(f"Created {OUTPUT.name} ({len(cells)} cells; training disabled).")
    return OUTPUT


if __name__ == "__main__":
    build()
