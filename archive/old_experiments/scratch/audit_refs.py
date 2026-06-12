import os
import re
from pathlib import Path

def audit_refs():
    project_dir = Path("c:/Users/user/Desktop/KTU intern/sciml models/paper_project")
    tex_files = list(project_dir.glob("**/*.tex"))
    
    label_pattern = re.compile(r'\\label\{([^}]+)\}')
    ref_pattern = re.compile(r'\\(ref|autoref)\{([^}]+)\}')
    img_pattern = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
    table_pattern = re.compile(r'\\begin\{(table|table\*)\}')
    figure_pattern = re.compile(r'\\begin\{(figure|figure\*)\}')
    
    labels = {}
    refs = []
    figures = []
    tables = []
    
    for tf in tex_files:
        rel_path = tf.relative_to(project_dir)
        with open(tf, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # strip comments
        content_clean = re.sub(r'(?<!\\)%.*$', '', content, flags=re.MULTILINE)
        
        # labels
        for match in label_pattern.finditer(content_clean):
            lbl = match.group(1)
            # Find context
            start = max(0, match.start() - 150)
            end = min(len(content_clean), match.end() + 150)
            context = content_clean[start:end].replace('\n', ' ')
            labels[lbl] = (rel_path, context)
            
        # refs
        for match in ref_pattern.finditer(content_clean):
            ref_type = match.group(1)
            ref_key = match.group(2)
            line_no = content_clean[:match.start()].count('\n') + 1
            refs.append((rel_path, line_no, ref_type, ref_key))
            
        # figures
        for match in img_pattern.finditer(content_clean):
            img_path = match.group(1)
            line_no = content_clean[:match.start()].count('\n') + 1
            figures.append((rel_path, line_no, img_path))
            
    print("=== ALL LABELS FOUND ===")
    for lbl, (path, ctx) in sorted(labels.items()):
        print(f"{lbl} in {path}")
        
    print("\n=== ALL REFERENCES TO AUDIT ===")
    for path, line, rtype, rkey in sorted(refs):
        resolved = "OK" if rkey in labels else "BROKEN"
        print(f"[{resolved}] {path}:L{line}: \\{rtype}{{{rkey}}}")
        
    print("\n=== ALL FIGURES USED ===")
    for path, line, img in sorted(figures):
        print(f"{path}:L{line}: {img}")

if __name__ == "__main__":
    audit_refs()
