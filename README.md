# Finnish ModernBERTs
- This repository contains most of the code used for the Finnish ModernBERT development
- for more details refer to the [article]()
- `src/data-tools` &rarr; code for basic data processing
    - The code for [Deduplication](https://github.com/ChenghaoMou/text-dedup), [PII-removal](https://github.com/mmanteli/multilingual-PII-tool), and [WikiExtrator](https://github.com/attardi/wikiextractor) are not hosted here, please refer 
     directly to sources.
- `src/finetuning` &rarr; code for finetuning models for retrieval evaluation
- `src/tokenizer` &rarr; code for tokenizer training   
- `src/training` &rarr; code for model training
- `src/scripts` &rarr; example scripts on running training on [LUMI](https://lumi-supercomputer.eu/)

# Reference
If you find this code useful, please refer using following citation:

~~~
@article{finnish-modernberts,
  author = {Akseli Reunamo and Laura-Maria Peltonen and Hans Moen and Sampo Pyysalo},
  title = {Pretraining Finnish ModernBERTs},
  year = {2025},
  journal={arXiv preprint },
    url={}, 
}
~~~
