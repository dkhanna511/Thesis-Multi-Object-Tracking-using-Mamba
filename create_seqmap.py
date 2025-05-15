import os

def write_folder_names_to_file(directory, output_file):
    # Get all folder names in the directory
    folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

    # Write to the file
    with open(output_file, "w") as file:
        file.write("name\n")  # Write the header
        file.writelines(f"{folder}\n" for folder in folders)

    print(f"Folder names written to '{output_file}'.")

# Example usage
source_directory = "datasets/VIP-HTD/train/"
output_text_file = "datasets/VIP-HTD/hockey-train.txt"

write_folder_names_to_file(source_directory, output_text_file)



source_directory = "datasets/VIP-HTD/val/"
output_text_file = "datasets/VIP-HTD/hockey-val.txt"

write_folder_names_to_file(source_directory, output_text_file)




source_directory = "datasets/VIP-HTD/test/"
output_text_file = "datasets/VIP-HTD/hockey-test.txt"

write_folder_names_to_file(source_directory, output_text_file)
