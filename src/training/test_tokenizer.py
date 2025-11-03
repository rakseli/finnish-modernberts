import argparse
from transformers import AutoTokenizer
# Parse arguments
parser = argparse.ArgumentParser(description="Test tokenizers")
parser.add_argument("--tokenizer", type=str, default="/scratch/<project_number>/avirtanen/finnish-modernberts-tokenizers/uniform/v1", help="path or name of tokenizer")

if __name__ == "__main__":
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    texts = ["Otetaas testiä miltä tokenisointi näyttää tällä tokenizerilla", "toinen lause"]
    max_length = 3
    tokens=tokenizer(texts, truncation=True, padding="max_length", max_length=max_length,return_overflowing_tokens=True)
    print("Tokens:", tokens)
    print("Token IDs:", tokens["input_ids"])
    decoded_text = tokenizer.decode(tokens['input_ids'][0], skip_special_tokens=True)
    print("Original Text:", texts[0])
    print("Decoded Text:", decoded_text)

