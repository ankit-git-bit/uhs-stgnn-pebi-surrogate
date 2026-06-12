import os
import re

def validate_latex_project(root_dir):
    print("==========================================")
    print("STARTING LATEX PROJECT SYNTAX VALIDATION")
    print("==========================================")
    
    main_tex_path = os.path.join(root_dir, "main.tex")
    if not os.path.exists(main_tex_path):
        print(f"Error: main.tex not found at {main_tex_path}")
        return False

    # Regex patterns
    input_pattern = re.compile(r'\\input\{([^}]+)\}')
    label_pattern = re.compile(r'\\label\{([^}]+)\}')
    ref_pattern = re.compile(r'\\ref\{([^}]+)\}')
    cite_pattern = re.compile(r'\\cite\{([^}]+)\}')
    includegraphics_pattern = re.compile(r'\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}')
    bib_key_pattern = re.compile(r'@\w+\{\s*([^,]+),')

    # Project databases
    files_to_read = [main_tex_path]
    read_files = set()
    
    labels = set()
    refs = []
    cites = []
    figures_used = []
    inputs_found = []

    # Read and parse tex files recursively
    while files_to_read:
        current_file = files_to_read.pop(0)
        if current_file in read_files:
            continue
        
        # Determine path of the file
        resolved_path = current_file
        if not os.path.exists(resolved_path):
            # Try adding .tex extension
            if os.path.exists(resolved_path + ".tex"):
                resolved_path += ".tex"
            else:
                print(f"Warning: Inputted file does not exist: {current_file}")
                continue

        read_files.add(resolved_path)
        
        try:
            with open(resolved_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception as e:
            print(f"Error reading file {resolved_path}: {e}")
            continue

        # Extract features
        # Remove comments first to avoid false matches
        content_no_comments = re.sub(r'(?<!\\)%.*$', '', content, flags=re.MULTILINE)

        # Labels
        for lbl in label_pattern.findall(content_no_comments):
            labels.add(lbl)
        
        # Refs
        for rf in ref_pattern.findall(content_no_comments):
            refs.append((rf, resolved_path))
            
        # Cites
        for ct in cite_pattern.findall(content_no_comments):
            # Citations can be comma separated, e.g. \cite{key1,key2}
            for part in ct.split(','):
                cites.append((part.strip(), resolved_path))
                
        # Image inclusions
        for img in includegraphics_pattern.findall(content_no_comments):
            figures_used.append((img, resolved_path))

        # Inputs/Includes
        for inp in input_pattern.findall(content_no_comments):
            # Resolve relative path from root_dir
            inp_path = os.path.join(root_dir, inp)
            inputs_found.append((inp, resolved_path))
            files_to_read.append(inp_path)

    # Load BibTeX keys
    bib_path = os.path.join(root_dir, "references.bib")
    bib_keys = set()
    if os.path.exists(bib_path):
        try:
            with open(bib_path, "r", encoding="utf-8") as f:
                bib_content = f.read()
            for key in bib_key_pattern.findall(bib_content):
                bib_keys.add(key.strip())
        except Exception as e:
            print(f"Error reading references.bib: {e}")
    else:
        print("Warning: references.bib not found!")

    # Check for errors
    errors_found = False
    
    print("\n--- Checking Input Files ---")
    for inp, parent in inputs_found:
        inp_full = os.path.join(root_dir, inp)
        if not os.path.exists(inp_full) and not os.path.exists(inp_full + ".tex"):
            print(f"ERROR: Input file '{inp}' referenced in '{os.path.basename(parent)}' does not exist.")
            errors_found = True
        else:
            print(f"OK: Input file '{inp}' found.")

    print("\n--- Checking Figure Files ---")
    for img, parent in figures_used:
        # Check in the root/figures directory
        img_full_path = os.path.join(root_dir, img)
        if not os.path.exists(img_full_path):
            # Maybe it needs extension
            has_ext = False
            for ext in ['.png', '.jpg', '.pdf', '.jpeg']:
                if os.path.exists(img_full_path + ext):
                    has_ext = True
                    break
            if not has_ext:
                print(f"ERROR: Figure file '{img}' referenced in '{os.path.basename(parent)}' does not exist.")
                errors_found = True
            else:
                print(f"OK: Figure file '{img}' (with extension) found.")
        else:
            print(f"OK: Figure file '{img}' found.")

    print("\n--- Checking Cross-References (labels vs refs) ---")
    for rf, parent in refs:
        if rf not in labels:
            print(f"ERROR: Reference '{rf}' in '{os.path.basename(parent)}' is undefined (no matching \\label found).")
            errors_found = True
        else:
            print(f"OK: Reference '{rf}' resolved.")

    print("\n--- Checking Citations (cites vs bibliography) ---")
    for ct, parent in cites:
        if ct not in bib_keys:
            print(f"ERROR: Citation '{ct}' in '{os.path.basename(parent)}' is not defined in references.bib.")
            errors_found = True
        else:
            print(f"OK: Citation '{ct}' resolved in references.bib.")

    print("\n--- Summary ---")
    print(f"Total Unique Labels found: {len(labels)}")
    print(f"Total References checked: {len(refs)}")
    print(f"Total Citations checked: {len(cites)}")
    print(f"Total Figures checked: {len(figures_used)}")
    print(f"Total Input files resolved: {len(read_files)}")
    
    if errors_found:
        print("\nResult: VALIDATION FAILED with errors! Check the output above.")
        return False
    else:
        print("\nResult: VALIDATION SUCCESSFUL! The project is compile-ready.")
        return True

if __name__ == "__main__":
    import sys
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    project_dir = os.path.join(project_root, "paper_project")
    if len(sys.argv) > 1:
        project_dir = sys.argv[1]
    validate_latex_project(project_dir)
