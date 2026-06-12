from pathlib import Path

def search_raw_eqs():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    targets = ["Eq. 4", "Eq. 5", "Eq. 9", "Eq. 12", "Eq. 7"]
    print("Searching for raw equation strings...")
    found_any = False
    for tf in tex_files:
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
        lines = content.splitlines()
        for idx, line in enumerate(lines):
            for t in targets:
                if t in line:
                    print(f"{tf.relative_to(project_dir)}:L{idx+1}: {line.strip()}")
                    found_any = True
    if not found_any:
        print("No raw equation strings found.")

if __name__ == "__main__":
    search_raw_eqs()
