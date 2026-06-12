import os
import shutil

src_root = "paper_project"
dest_roots = ["uhs_trial_1", "uha_trial_1"]

# Folders to synchronize
folders = ["sections", "supplementary", "figures", "tables"]

# Files to copy directly
files = [
    "main.tex",
    "references.bib",
    "reviewer2_report.md",
    "publication_readiness.md",
    "pdf_quality_report.md",
    "story_audit.md",
    "conference_questions.md",
    "future_work_roadmap.md",
    "corrections_log.md"
]

for dest_root in dest_roots:
    print(f"\nSynchronizing {src_root} -> {dest_root}...")
    os.makedirs(dest_root, exist_ok=True)
    
    # Copy directories
    for folder in folders:
        src_dir = os.path.join(src_root, folder)
        dest_dir = os.path.join(dest_root, folder)
        
        if os.path.exists(src_dir):
            os.makedirs(dest_dir, exist_ok=True)
            for item in os.listdir(src_dir):
                src_item = os.path.join(src_dir, item)
                dest_item = os.path.join(dest_dir, item)
                
                if os.path.isfile(src_item):
                    shutil.copy2(src_item, dest_item)
                    print(f"  [{dest_root}] Copied dir file: {src_item} -> {dest_item}")
                elif os.path.isdir(src_item):
                    shutil.copytree(src_item, dest_item, dirs_exist_ok=True)
                    print(f"  [{dest_root}] Copied subdirectory: {src_item} -> {dest_item}")
        else:
            print(f"  [{dest_root}] Warning: Source directory {src_dir} does not exist")

    # Copy direct files
    for file in files:
        src_file = os.path.join(src_root, file)
        dest_file = os.path.join(dest_root, file)
        
        if os.path.exists(src_file):
            shutil.copy2(src_file, dest_file)
            print(f"  [{dest_root}] Copied file: {src_file} -> {dest_file}")
        else:
            print(f"  [{dest_root}] Warning: Source file {src_file} does not exist")

print("\nSynchronization complete for all destinations!")
