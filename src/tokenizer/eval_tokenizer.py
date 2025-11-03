import sys
import os
import json
import regex
import tiktoken
from statistics import mean
from argparse import ArgumentParser
from transformers import AutoTokenizer
WORD_RE = regex.compile(r'[[:alnum:]]+|[^[:space:]]')


def make_tokenizer(name):
  if name == 'gpt-4o':
    enc = tiktoken.encoding_for_model(name)
    return lambda t: enc.encode(t)
  else:
    tokenizer = AutoTokenizer.from_pretrained(
      name,
      trust_remote_code=True,
    )
    return lambda t: tokenizer(t).input_ids
 
if __name__ == '__main__':
  ap = ArgumentParser()
  ap.add_argument('--whitespace', action='store_true',
          help='split words by whitespace')
  ap.add_argument('--output_path',type=str)
  ap.add_argument('--tokenizer')
  ap.add_argument('--files', nargs='+')
  args = ap.parse_args()
  tokenizer = make_tokenizer(args.tokenizer)
  fertilities = []
  total_token_count, total_word_count = 0, 0
  results = {}
  for fn in args.files:
    lines = []
    with open(fn) as f:
      for line in f:
        row = json.loads(line)
        lines.append(line) 

      token_count, word_count = 0, 0
      for line in lines:
        token_count += len(tokenizer(line))
        if args.whitespace:
          word_count += len(line.split())
        else:
          word_count += len(WORD_RE.findall(line))
      print(f'{os.path.basename(fn)} {token_count}/{word_count} '
         f'({token_count/word_count:.4f})')
      results[os.path.basename(fn)]=token_count/word_count
      fertilities.append(token_count/word_count)
      total_token_count += token_count
      total_word_count += word_count
  results['total']=total_token_count/total_word_count
  print(f'TOTAL {total_token_count}/{total_word_count} '
     f'({total_token_count/total_word_count:.2f})')
  print(f'AVERAGE {mean(fertilities):.2f}')
  results['average']=mean(fertilities)
  with open(f"{args.output_path}/{os.path.basename(args.tokenizer)}_fertilities.json", "w") as f:
    json.dump(results,f,ensure_ascii=False)