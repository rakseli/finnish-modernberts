import json
import argparse
import os

def normalize_jsonl(input_path):
    # Create output filename by adding "_fixed" before the extension
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_fixed{ext}"

    with open(input_path, "r", encoding="utf-8") as infile, open(output_path, "w", encoding="utf-8") as outfile:
        for line in infile:
            try:
                # First decode outer string
                decoded_line = json.loads(line.strip())
                # Then parse it as JSON
                if not isinstance(decoded_line,dict):
                    json_obj = json.loads(decoded_line)
                else:
                    json_obj = decoded_line
                # Write to output
                json.dump(json_obj, outfile, ensure_ascii=False)
                outfile.write("\n")
            except json.JSONDecodeError as e:
                print(f"Skipping invalid line due to error: {e}")

    print(f"Normalized file saved as: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fix double-encoded JSONL file.")
    parser.add_argument("--input_file", help="Path to the weird-format JSONL file")
    args = parser.parse_args()

    normalize_jsonl(args.input_file)