from pathlib import Path
import shutil
import xml.etree.ElementTree as ET
import argparse

def load_config(config_path):
    tree = ET.parse(config_path)
    root = tree.getroot()
    required = [f.text.strip() for f in root.findall("./required_files/file")]
    return {"required_files": required}

def main():
    parser = argparse.ArgumentParser(description='Build autograder package for Gradescope')
    parser.add_argument('--config', type=str, default='autograder_config.xml',
                        help='Path to autograder config file (default: autograder_config.xml)')
    args = parser.parse_args()
    
    lab_dir = Path.cwd()
    config_path = lab_dir / args.config

    # Shared autograder files
    shared = Path(__file__).resolve().parent.parent.parent / "gradescope"

    build_dir = lab_dir / "autograder_build"
    if build_dir.exists():
        shutil.rmtree(build_dir)
    build_dir.mkdir()

    # Copy shared files
    for f in ["autograde.py", "run_autograder", "setup.sh", "requirements.txt"]:
        shutil.copy(shared / f, build_dir / f)

    # Copy config
    shutil.copy(config_path, build_dir / "autograder_config.xml")

    # Zip
    shutil.make_archive("autograder", "zip", build_dir)

    print("autograder.zip created successfully.")