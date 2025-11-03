import torch
import argparse
import os
from transformers import FillMaskPipeline,AutoTokenizer,AutoConfig,AutoModelForMaskedLM



def load_checkpoint(checkpoint_path):
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", default=None, help="Path to checkpoint")
    args = parser.parse_args()
    checkpoint = load_checkpoint(f"{args.checkpoint_path}/model.bin")
    if checkpoint['config']['model_size'] == 'base' or checkpoint['config']['model_size'] == 'tiny':
        model_id = "answerdotai/ModernBERT-base"
    else:
        model_id = "answerdotai/ModernBERT-large"
    print()
    tokenizer = AutoTokenizer.from_pretrained(checkpoint['config']["tokenizer_path"])
    vocab_size = len(tokenizer)
    model_config = AutoConfig.from_pretrained(model_id)
    model_config.local_rope_theta = checkpoint['config']["local_rope_theta"]
    model_config.local_attention = checkpoint['config']["window_size"]
    model_config.vocab_size = vocab_size
    model_config.sparse_pred_ignore_index=-100
    model_config.deterministic_flash_attn=True
    model_config.pad_token_id = tokenizer.pad_token_id
    model_config.eos_token_id = tokenizer.sep_token_id
    model_config.sep_token_id = tokenizer.sep_token_id
    model_config.bos_token_id = tokenizer.cls_token_id
    model_config.cls_token_id = tokenizer.cls_token_id
    
    model_config.max_position_embeddings = checkpoint["config"]["current_max_length"]
    model_config.global_rope_theta = checkpoint["config"]["current_global_rope_theta"]
    if checkpoint["config"]['model_size'] == 'tiny':
        model_config.num_hidden_layers = 6

    print("Starting loading model weights...",flush=True)
    model = AutoModelForMaskedLM.from_config(config=model_config)
    model.load_state_dict(checkpoint["model"],strict=True)
    
    #model= AutoModel.from_pretrained(pretrained_model_name_or_path=None,config=model_config,state_dict=checkpoint['model'])
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    sentences = ["Turku on suomen [MASK].","Vi [MASK] Helsingförs störta","Where are [MASK] at?","Mihin kanttaa mennä jos haluaa rahaa ilmaiseksi? Luulisin [MASK] voisi olla hyvä."]
    inputs = tokenizer("Turku on suomen [MASK]", return_tensors="pt").to(device)
    model.to(device)
    model.eval()
    with torch.no_grad():
        res = model(**inputs)
    mask_token_index = (inputs.input_ids == tokenizer.mask_token_id)[0].nonzero(as_tuple=True)[0]
    predicted_token_id = res.logits[0, mask_token_index].argmax(axis=-1)
    print(tokenizer.decode(predicted_token_id.item()))
    fill_masker = FillMaskPipeline(model=model,tokenizer=tokenizer,device=device)
    for s in sentences:
        print(fill_masker(s),flush=True)
    print("Model weights loaded",flush=True)
    save_dir = os.path.dirname(args.checkpoint_path)
    save_name = os.path.basename(args.checkpoint_path)
    full_save_path = f"{save_dir}/hf-version-{save_name}"
    print(f"Saving model into {full_save_path}...",flush=True)
    #print(f"Model before saving: {model}")
    model.save_pretrained(full_save_path)
    del model
    print("Validation conversion...",flush=True)
    model = AutoModelForMaskedLM.from_pretrained(full_save_path)
    # Load both state_dicts
    ckpt1 = torch.load(f"{args.checkpoint_path}/model.bin", map_location="cpu")
    ckpt2 = model.state_dict()

    # If these are full checkpoints (with extra keys), extract the state_dicts
    ckpt1 = ckpt1['model']
    ckpt2 = ckpt2.get("state_dict", ckpt2)

    # Compare keys
    ckpt1_keys = set(ckpt1.keys())
    ckpt2_keys = set(ckpt2.keys())

    only_in_ckpt1 = ckpt1_keys - ckpt2_keys
    only_in_ckpt2 = ckpt2_keys - ckpt1_keys
    common_keys = ckpt1_keys & ckpt2_keys

    if only_in_ckpt1:
        print("Keys only in original checkpoint:", only_in_ckpt1)
    if only_in_ckpt2:
        print("Keys only in Hugging Face checkpoint:", only_in_ckpt2)

    # Compare parameter values for common keys
    for key in common_keys:
        if not torch.allclose(ckpt1[key], ckpt2[key], atol=1e-6):
            print(f"Mismatch in weight for key: {key}")

