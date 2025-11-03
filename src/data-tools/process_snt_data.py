import json
import re
import glob
import os
import concurrent.futures

def parse_tags_to_jsonl(text_file,parent_folfer,f_num):
    success={'res':True}
    heading_re = re.compile(r'<head>(.*?)<\/head>', re.DOTALL)
    p_re = re.compile(r'<p>', re.DOTALL)
    div0_type_re = re.compile(r'<div0 type=["\'](.*?)["\']>', re.DOTALL)
    try:
        with open(text_file,"r",encoding='utf-8',errors='replace') as input_file:
            text = input_file.readlines()
    except Exception as e:
        success['res']=False
        success['error']=e
        return success
    # Find all sections
    paragraphs = []
    t_type = None
    first=True
    d_id = 0
    data = []
    for index,l in enumerate(text):
        type_match = div0_type_re.match(l)
        if type_match:
            t_type = type_match.group(1)
            continue
        h_match = heading_re.match(l)
        if h_match:
            if first:
                heading = h_match.group(0)
                heading = re.sub(r'<.*?>', '',heading).strip()
                heading = re.sub(r'\||}|{|\*', lambda match: {'|': 'ö', '}': 'å', '{': 'ä','*':''}[match.group(0)], heading)
                first=False
                continue
            else:
                t = "\n ".join(paragraphs)
                t = re.sub(r'<[^>]+>', '',t)
                result = re.sub(r'\||}|{|\*', lambda match: {'|': 'ö', '}': 'å', '{': 'ä','*':''}[match.group(0)], t)
                json_obj = {
                         "id": f"fstc-{parent_folfer}-{f_num}-{d_id}",
                         "heading": heading,
                         "type": t_type,
                         "text": result
                            }
                data.append(json_obj)
                d_id += 1
                heading = h_match.group(0)
                heading = re.sub(r'<.*?>', '',heading).strip()
                heading = re.sub(r'\||}|{|\*', lambda match: {'|': 'ö', '}': 'å', '{': 'ä','*':''}[match.group(0)], heading)

                paragraphs = []
                continue
                
        p_match = p_re.match(l)
        if p_match:
            paragraphs.append(text[index+1])
        
    with open(f'/scratch/<project_number>/<user_name>/finnish-modernberts/data/sv/fstc-cleaned/{parent_folfer}-{f_num}.jsonl', 'w') as outfile:
        for entry in data:
            json_line = json.dumps(entry,ensure_ascii=False)
            outfile.write(json_line+'\n')

    return success

if __name__ == "__main__":
    snt_files = glob.glob("/scratch/<project_number>/<user_name>/finnish-modernberts/data/sv/fstc-source/originals/*/*.snt",recursive=True)
    parent_folders = [os.path.basename(os.path.dirname(file)) for file in snt_files]
    with concurrent.futures.ProcessPoolExecutor(max_workers=10) as executor:
        # Submit tasks individually
        futures = [executor.submit(parse_tags_to_jsonl,f,p,i) for i,(f,p) in enumerate(zip(snt_files,parent_folders),start=1)]
        results = [f.result() for f in concurrent.futures.as_completed(futures)]
        
    print(f"Results: {results}")