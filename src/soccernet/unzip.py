import os
import zipfile

def unzip_soccerNet(localPath):
    """
    Recursively finds and unzips all .zip files within the given localPath.
    Each zip file is extracted into a directory with the same name as the zip file (without .zip extension)
    in the same location as the zip file.
    """
    for root, dirs, files in os.walk(localPath):
        for file in files:
            if file.endswith(".zip"):
                zip_file_path = os.path.join(root, file)
                # Create output directory name by removing .zip extension
                output_dir_name = os.path.splitext(zip_file_path)[0]
                
                # Create the output directory if it doesn't exist
                if not os.path.exists(output_dir_name):
                    os.makedirs(output_dir_name)
                
                try:
                    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
                        zip_ref.extractall(output_dir_name)
                    print(f"Successfully unzipped {zip_file_path} to {output_dir_name}")
                except zipfile.BadZipFile:
                    print(f"Error: {zip_file_path} is a bad zip file and could not be unzipped.")
                except Exception as e:
                    print(f"An error occurred while unzipping {zip_file_path}: {e}")

if __name__ == '__main__':
    # Example usage:
    # Replace '/path/to/your/downloads' with the actual path you want to process
    # For testing, you might want to create a dummy directory with some zip files.
    # e.g. local_directory_to_scan = 'dataset/SoccerNet/'
    # Make sure this path exists and contains zip files for testing.
    
    # To run this script directly for testing:
    # 1. Create a directory, e.g., 'test_zip_dir'
    # 2. Put some .zip files into 'test_zip_dir' and its subdirectories.
    # 3. Change 'path_to_scan' to 'test_zip_dir'
    # 4. Run `python src/soccernet/unzip.py` from the root of your project.

    # path_to_scan = "dataset/SoccerNet/" # Modify this path for your testing
    # print(f"Starting to unzip files in {os.path.abspath(path_to_scan)}...")
    # unzip_soccerNet(path_to_scan)
    # print("Unzipping process completed.")
    pass # Placeholder for example usage if run directly
