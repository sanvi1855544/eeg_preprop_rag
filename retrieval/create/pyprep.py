import os
import ast
from tqdm import tqdm
from sklearn.model_selection import train_test_split  # add this import
from .utils import save_file_jsonl, save_tsv_dict

def extract_functions_from_repo(repo_path):
    data = []
    for root, dirs, files in os.walk(repo_path):
        for file in files:
            if file.endswith(".py"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r", encoding="utf-8") as f:
                        source = f.read()
                    tree = ast.parse(source)
                    for node in ast.walk(tree):
                        if isinstance(node, ast.FunctionDef):
                            docstring = ast.get_docstring(node)
                            code = ast.unparse(node) if hasattr(ast, "unparse") else source[node.lineno - 1: node.end_lineno]
                            data.append({
                                "repository_name": "pyprep",
                                "func_path_in_repository": os.path.relpath(file_path, repo_path).replace("/", "_"),
                                "func_name": node.name,
                                "func_documentation_string": docstring or "",
                                "func_code_string": code
                            })
                except Exception as e:
                    print(f"Skipping {file_path}: {e}")
    return data

def document2code(data):
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
    repo_path = "pyprep/pyprep"  # Adjust if needed
    output_dir = "datasets/pyprep_dataset"
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "qrels"), exist_ok=True)

    print("Extracting functions...")
    extracted = extract_functions_from_repo(repo_path)

    print("Processing into IR format...")
    queries, docs, qrels = document2code(extracted)

    # Save corpus and queries as usual
    save_file_jsonl(queries, os.path.join(output_dir, "queries.jsonl"))
    save_file_jsonl(docs, os.path.join(output_dir, "corpus.jsonl"))

    # Split qrels into train/val/test (80/10/10 split)
    train_qrels, temp_qrels = train_test_split(qrels, test_size=0.2, random_state=42)
    val_qrels, test_qrels = train_test_split(temp_qrels, test_size=0.5, random_state=42)

    # Save splits
    save_tsv_dict(train_qrels, os.path.join(output_dir, "qrels", "train.tsv"), ["query-id", "corpus-id", "score"])
    save_tsv_dict(val_qrels, os.path.join(output_dir, "qrels", "val.tsv"), ["query-id", "corpus-id", "score"])
    save_tsv_dict(test_qrels, os.path.join(output_dir, "qrels", "test.tsv"), ["query-id", "corpus-id", "score"])

if __name__ == "__main__":
    main()
