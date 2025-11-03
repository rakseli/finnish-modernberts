
import argparse
import os
from transformers import PreTrainedTokenizerFast,AutoTokenizer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_path", type=str, help="tokenizer path")
    parser.add_argument("--save_path",type=str,default="/scratch/<project_number>/<user_name>/finnish-modernberts/results/tokenizers",help="save path")
    parser.add_argument("--force",action="store_true")
    args = parser.parse_args()
    tok = AutoTokenizer.from_pretrained(
                                        pretrained_model_name_or_path=args.tokenizer_path,
                                        clean_up_tokenization_spaces=True, 
                                        pad_token="[PAD]",
                                        unk_token="[UNK]",
                                        cls_token="[CLS]", 
                                        sep_token="[SEP]",
                                        mask_token="[MASK]",
                                        model_max_length=128000,
                                        add_prefix_space=False,
                                        model_input_names = ["input_ids", "attention_mask"]
                                        )
    tok.save_pretrained(f"{args.save_path}/{os.path.basename(args.tokenizer_path)}_pretrained")
    tok = PreTrainedTokenizerFast.from_pretrained(f"{args.save_path}/{os.path.basename(args.tokenizer_path)}_pretrained")
    print("Done saving",flush=True)
    print("Validating conversion and examing tokenizer outputs...")
    og_tokenizer = PreTrainedTokenizerFast.from_pretrained("answerdotai/ModernBERT-base")
    sami_string = "Oaidnit tabeallas ahte leat unnán ohccit ja unnán oahppit geat duođaid čađahit skuvlaoahpu duojis. Dán skuvlajahkái ii lean oktage ohcci design ja duodjái."
    emoji_string = """🔥➡️🪵🛖➡️🕯️🔥➡️🌡️📈➡️💦🪣➡️😌♨️"""
    fi_string = "Turkua on perinteisesti pidetty Suomen porttina länteen. Aurajoen suulle syntynyt kaupunki on aina ollut merkittävä kaupallinen satamakaupunki."
    sv_string = "Huset Montfort\n\nHuset Montfort av Bretagne regerade i hertigdömet Bretagne från 1365 till 1514."
    test_sentences = [sami_string,emoji_string,fi_string,sv_string]
    for s in test_sentences:
        hf_tokenization = tok.tokenize(s)
        hf_encoding = tok.encode(s)
        m_tokenization = og_tokenizer.tokenize(s)
        m_encoding = og_tokenizer.encode(s)
        print(f"Orginal sentence: {s}")
        print(f"New tokenizer tokenization: {hf_tokenization}")
        print(f"New tokenizer encoding: {hf_encoding}")
        print(f"New tokenizer decoding: {tok.decode(hf_encoding)}")
        print(f"ModernBert tokenizer tokenization: {m_tokenization}")
        print(f"ModernBert tokenizer encoding: {m_encoding}")
        print(f"ModernBert tokenizer decoding: {og_tokenizer.decode(m_encoding)}")

