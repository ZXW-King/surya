import os

from huggingface_hub import snapshot_download

def get_model(save_path_list):
    for save_path in save_path_list:
        if not os.path.exists(save_path):
            os.makedirs(save_path,exist_ok=True)
        try:
            # 下载指定仓库
            local_dir = snapshot_download(repo_id=save_path, revision="main", local_dir=save_path)
            print(f"Files downloaded to: {local_dir}")
        except:
            continue

if __name__ == '__main__':
    path_list = ["vikp/surya_det3",'datalab-to/surya_layout',"vikp/surya_rec2","vikp/surya_tablerec","datalab-to/ocr_error_detection"]
    get_model(path_list)