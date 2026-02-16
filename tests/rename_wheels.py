import argparse
import glob
import os
import sys

def rename_wheels(dist_dir, dbr_version):
    if not os.path.exists(dist_dir):
        print(f"Warning: Directory '{dist_dir}' does not exist. Skipping wheel rename.")
        return

    # Tags to replace, prioritizing newer/specific ones
    tags_to_replace = [
        "manylinux_2_39",
        "manylinux_2_35",
        "manylinux_2_34",
        "manylinux_2_31",
        "manylinux_2_28",
        "manylinux_2_24",
        "manylinux2014",
        "manylinux2010",
        "manylinux1"
    ]
    
    replacement = f"dbr_{dbr_version}"
    
    wheels = glob.glob(os.path.join(dist_dir, "*.whl"))
    for whl in wheels:
        dirname, filename = os.path.split(whl)
        new_filename = filename
        
        for tag in tags_to_replace:
            if tag in filename:
                new_filename = filename.replace(tag, replacement)
                break
        
        if new_filename != filename:
            print(f"Renaming wheel for Databricks: {filename} -> {new_filename}")
            os.rename(whl, os.path.join(dirname, new_filename))

def main():
    parser = argparse.ArgumentParser(description="Rename wheels for Databricks deployment.")
    parser.add_argument("--dbr", action="store_true", help="Rename manylinux tags to dbr tags")
    parser.add_argument("--dbr-version", default="17", help="Databricks Runtime version (default: 17)")
    parser.add_argument("--dist-dir", default="dist", help="Directory containing wheels")

    args = parser.parse_args()

    if args.dbr:
        rename_wheels(args.dist_dir, args.dbr_version)

if __name__ == "__main__":
    main()