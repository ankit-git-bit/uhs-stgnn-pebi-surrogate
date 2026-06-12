import os
import re

def main():
    print("--- Searching for graphics in paper_project/sections ---")
    sections_dir = 'paper_project/sections'
    tex_files = [f for f in os.listdir(sections_dir) if f.endswith('.tex')]
    
    fig_refs = []
    for tf in tex_files:
        path = os.path.join(sections_dir, tf)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            # find all \includegraphics[...]{...} or \includegraphics{...}
            matches = re.findall(r'\\includegraphics(?:\[.*?\])?\{(.*?)\}', content)
            if matches:
                print(f"File: {tf}")
                for m in matches:
                    print(f"  - {m}")
                    fig_refs.append(m)
                    
    print("\n--- Searching for PIGNN in paper_project/sections ---")
    pignn_matches = []
    for tf in tex_files:
        path = os.path.join(sections_dir, tf)
        with open(path, 'r', encoding='utf-8') as f:
            content = f.read()
            matches = re.findall(r'(?i)pignn', content)
            if matches:
                print(f"File {tf} has {len(matches)} occurrences of PIGNN.")
                pignn_matches.extend(matches)
                
    print(f"Total PIGNN occurrences in sections: {len(pignn_matches)}")

if __name__ == "__main__":
    main()
