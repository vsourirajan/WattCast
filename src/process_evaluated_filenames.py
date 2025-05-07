import re

# Function to extract processed file names from log file
def extract_processed_files(log_file_path, output_file_path):
    processed_files = []

    # Regular expression to match "Processing file:" lines
    file_pattern = re.compile(r"Processing file: (.+)")

    # Read log file and extract processed file names
    with open(log_file_path, 'r') as log_file:
        for line in log_file:
            match = file_pattern.search(line)
            if match:
                processed_files.append(match.group(1))

    # Write the processed file names to output file
    with open(output_file_path, 'w') as output_file:
        for file in processed_files:
            output_file.write(file + '\n')

    print(f"Extracted {len(processed_files)} processed files.")

# Usage example (adjust file paths as needed)
if __name__ == "__main__":
    log_file_path = "./lstm_training.log"  # Replace with your log file path
    output_file_path = "processed_files.txt"     # Output file name
    extract_processed_files(log_file_path, output_file_path)
