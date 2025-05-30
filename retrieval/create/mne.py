import argparse
import os
import ast
from pathlib import Path
from tqdm import tqdm
from create.utils import save_tsv_dict, save_file_jsonl


def extract_docs_and_code_from_mne(mne_source_dir):
    """Extracts function docstrings and source code from MNE .py files."""
    data = []
    mne_source_dir = Path(mne_source_dir)
    for py_file in mne_source_dir.rglob("*.py"):
        if "tests" in py_file.parts or py_file.name.startswith("test_"):
            continue  # skip test files

        try:
            with open(py_file, "r", encoding="utf-8") as f:
                source = f.read()
            tree = ast.parse(source, filename=str(py_file))
        except SyntaxError:
            continue  # skip malformed files

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                docstring = ast.get_docstring(node)
                if not docstring or node.name.startswith("_"):
                    continue  # skip private or undocumented functions

                try:
                    code_str = ast.get_source_segment(source, node)
                except:
                    code_str = None

                if code_str:
                    data.append({
                        "func_name": node.name,
                        "func_documentation_string": docstring.strip(),
                        "func_code_string": code_str.strip(),
                        "repository_name": "mne",
                        "func_path_in_repository": str(py_file.relative_to(mne_source_dir))
                    })
    return {"train": data}


def document2code(data, split="train"):
    data = data[split]
    queries = []
    docs = []
    qrels = []

    for item in tqdm(data):
        doc = item["func_documentation_string"]
        code = item["func_code_string"]
        doc_id = "{repository_name}_{func_path_in_repository}_{func_name}_doc".format_map(item)
        code_id = "{repository_name}_{func_path_in_repository}_{func_name}_code".format_map(item)
        queries.append({"_id": doc_id, "text": doc, "metadata": {}})
        docs.append({"_id": code_id, "title": item["func_name"], "text": code, "metadata": {}})
        qrels.append({"query-id": doc_id, "corpus-id": code_id, "score": 1})

    return queries, docs, qrels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_dir", type=str, required=True, help="Path to MNE source code (e.g., mne/)")
    parser.add_argument("--output_dir", type=str, default="datasets", help="Where to save the IR-formatted data")
    args = parser.parse_args()

    # Load and extract
    print(f"Extracting from MNE source: {args.source_dir}")
    dataset = extract_docs_and_code_from_mne(args.source_dir)

    # Prepare output
    path = os.path.join(args.output_dir, "mne_ir_dataset")
    os.makedirs(path, exist_ok=True)
    os.makedirs(os.path.join(path, "qrels"), exist_ok=True)

    # Convert and save
    queries, docs, qrels = document2code(dataset, split="train")
    save_file_jsonl(queries, os.path.join(path, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(path, "corpus.jsonl"))
    save_tsv_dict(qrels, os.path.join(path, "qrels", "test.tsv"), ["query-id", "corpus-id", "score"])

    print(f"Saved {len(queries)} queries, {len(docs)} docs, and {len(qrels)} qrels to: {path}")


if __name__ == "__main__":
    main()
