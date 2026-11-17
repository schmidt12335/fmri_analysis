"""Repository checks (also run by GitHub Actions): `python tools/check_repo.py`

1. every .py file is valid Python,
2. every notebook is valid JSON and every code cell is valid Python (IPython `%`/`!` lines are ignored),
3. notebooks contain no outputs or execution counts (outputs make the repository large and the diffs unreadable;
   run `nbstripout <notebook>` or install the git filter, see the README). `--allow-outputs` skips this check.

Exit code 1 if anything is wrong. Needs only the Python standard library."""
import argparse
import ast
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def repo_files(suffix):
    for path in sorted(ROOT.rglob(f"*{suffix}")):
        parts = path.relative_to(ROOT).parts
        if any(p.startswith(".") or p in ("venv", "env", "__pycache__") for p in parts) or path.name.startswith("._"):
            continue
        yield path


def code_without_magics(source):
    """Cell source with IPython line magics (%...) and shell escapes (!...) replaced by `pass`."""
    lines = source.splitlines()
    if lines and lines[0].lstrip().startswith("%%"):
        return ""
    return "\n".join(
        line[: len(line) - len(line.lstrip())] + "pass" if line.lstrip().startswith(("%", "!")) else line
        for line in lines
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--allow-outputs", action="store_true", help="do not fail on notebooks that contain outputs")
    args = parser.parse_args()

    errors = []

    for path in repo_files(".py"):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except SyntaxError as e:
            errors.append(f"{path.relative_to(ROOT)}: line {e.lineno}: {e.msg}")

    n_notebooks = 0
    for path in repo_files(".ipynb"):
        n_notebooks += 1
        name = path.relative_to(ROOT)
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except ValueError as e:
            errors.append(f"{name}: not valid JSON ({e})")
            continue
        n_outputs = 0
        for i, cell in enumerate(notebook.get("cells", [])):
            if cell.get("cell_type") != "code":
                continue
            try:
                ast.parse(code_without_magics("".join(cell.get("source", []))))
            except SyntaxError as e:
                errors.append(f"{name}: cell {i}: line {e.lineno}: {e.msg}")
            if cell.get("outputs") or cell.get("execution_count") is not None:
                n_outputs += 1
        if n_outputs and not args.allow_outputs:
            errors.append(f"{name}: {n_outputs} code cells have outputs or an execution count (run: nbstripout {name})")

    if errors:
        print(f"{len(errors)} problem(s):")
        for e in errors:
            print(f"  {e}")
        return 1
    print(f"OK: all Python files and {n_notebooks} notebooks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
