# Finnish ModernBERTs
- This repository contains most of the code used for the Finnish ModernBERT development
- for more details refer to the [article](https://arxiv.org/abs/2511.09213)
- `src/data-tools` &rarr; code for basic data processing
    - The code for [Deduplication](https://github.com/ChenghaoMou/text-dedup), [PII-removal](https://github.com/mmanteli/multilingual-PII-tool), and [WikiExtrator](https://github.com/attardi/wikiextractor) are not hosted here, please refer 
     directly to sources.
- `src/finetuning` &rarr; code for finetuning models for retrieval evaluation
- `src/tokenizer` &rarr; code for tokenizer training   
- `src/training` &rarr; code for model training
- `src/scripts` &rarr; example scripts on running training on [LUMI](https://lumi-supercomputer.eu/)

# Known limitations

The code has not been engineered or tested for general use. Expect to adapt paths, launchers, and
environments.

Known limitations of the code used for the released models:

**Pretraining (`src/training/`)**
- RoPE theta warmup: setting `rotary_emb.base` at runtime does not recompute
  the rotary frequencies. The new global theta (1M) takes effect only when the
  model is rebuilt from a checkpoint (on resume or HF export).
- `no_sync` wraps only the backward pass, so gradients are all-reduced on every
  micro-batch. The result is correct but slower.
- Logged train/validation accuracy and gradient norm are not correctly
  normalized. The logged loss is correct.
