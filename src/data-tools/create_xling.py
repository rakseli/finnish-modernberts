import json
import os
import glob
langs = ['en','se','fi']
instructions = {'se-en':'Jorgal eŋgelasgillii: ',
                'se-sv':'Jorgal ruoŧagillii: ',
                'se-fi':'Jorgal suomagillii: ',
                'en-se':'Translate into Northern Sámi: ',
                'sv-se':'Översätt till nordsamiska: ',
                'fi-se':'Käännä pohjoissaameksi: '
                }

if __name__ == "__main__":
    for l in langs:
        lang_files= glob.glob(f"/scratch/<project_number>/<user_name>/finnish-modernberts/data/xling/wikimedia.{l}-*")
        source = lang_files[0] if lang_files[0][-2:] == l else lang_files[1]
        target = lang_files[0] if lang_files[0][-2:] != l else lang_files[1]
        with open(source,"r") as f:
            source_lines = f.readlines()
        with open(target,"r") as f:
            target_lines = f.readlines()
        with open(f"/scratch/<project_number>/<user_name>/finnish-modernberts/data/xling/{source[-2:]}-{target[-2:]}.jsonl", 'w') as outfile:
            for s,t in zip(source_lines,target_lines):
                i = instructions[f"{source[-2:]}-{target[-2:]}"]
                t = i + s + "\n" + t
                d = {'text':t}
                json_line = json.dumps(d,ensure_ascii=False)
                outfile.write(json_line+'\n')
        with open(f"/scratch/<project_number>/<user_name>/finnish-modernberts/data/xling/{target[-2:]}-{source[-2:]}.jsonl", 'w') as outfile:
            for s,t in zip(source_lines,target_lines):
                i = instructions[f"{target[-2:]}-{source[-2:]}"]
                t = i + t + "\n" + s + "\n"
                d = {'text':t}
                json_line = json.dumps(d,ensure_ascii=False)
                outfile.write(json_line+'\n')