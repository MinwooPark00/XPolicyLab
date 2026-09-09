import os
import zipfile

def unzip_files(directory):
    # Loop through each file in the directory
    for filename in os.listdir(directory):
        # Check if the file starts with "part2" and is a zip file
        if filename.startswith("part2") and filename.endswith(".zip"):
            filepath = os.path.join(directory, filename)
            # Create a folder with the same name as the zip file (without .zip extension)
            extract_path = os.path.join(directory, filename[:-4])
            # Extract zip file contents
            with zipfile.ZipFile(filepath, 'r') as zip_ref:
                zip_ref.extractall(extract_path)
            print(f"Extracted {filename} to {extract_path}")

# Set the directory to your target folder
directory = '/path/to/your/directory'
unzip_files(directory)
