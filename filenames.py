import os

REPO_DIR = "./CBraMod"  # ← Replace this

for root, _, files in os.walk(REPO_DIR):
    for file in files:
        if file.endswith(".py"):
            rel_path = os.path.relpath(os.path.join(root, file), REPO_DIR)
            print(rel_path)
