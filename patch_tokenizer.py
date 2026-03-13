import json
import sys
from huggingface_hub import hf_hub_download

def patch_tokenizer(model_name):
    try:
        tok_cfg_path = hf_hub_download(model_name, "tokenizer_config.json")
        with open(tok_cfg_path) as f:
            tok_cfg = json.load(f)
        
        if tok_cfg.get("tokenizer_class") == "TokenizersBackend":
            print(f"⚠ Patching invalid tokenizer_class 'TokenizersBackend' → removing field")
            del tok_cfg["tokenizer_class"]
            with open(tok_cfg_path, "w") as f:
                json.dump(tok_cfg, f, indent=2)
            print(f"  Patched: {tok_cfg_path}")
            return True
        else:
            print(f"tokenizer_class is '{tok_cfg.get('tokenizer_class')}' — no patch needed")
            return False
    except Exception as e:
        print(f"Error processing {model_name}: {e}")
        return False

if __name__ == "__main__":
    model = sys.argv[1]
    if not model:
        print("You must include an HF repo!")
    else:
        patch_tokenizer(model)
