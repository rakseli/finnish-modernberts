from datasets import load_dataset,disable_caching,DownloadMode
disable_caching()
if __name__ == "__main__":
    dataset_train = load_dataset("rpa020/SALT-train",data_files="data-00000-of-00001.arrow",download_mode=DownloadMode.FORCE_REDOWNLOAD,split='train')
    dataset_train.to_json("/scratch/<project_number>/<user_name>/finnish-modernberts/data/sme/salt-train.jsonl",force_ascii=False)
