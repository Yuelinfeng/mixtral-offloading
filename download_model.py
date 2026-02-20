import os
from huggingface_hub import snapshot_download

# 模型仓库 ID
repo_id = "lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo"

# 本地保存目录 (与 benchmark.py 中的 state_path 一致)
local_dir = "Mixtral-8x7B-Instruct-v0.1-offloading-demo"

print(f"准备下载模型: {repo_id}")
print(f"保存目录: {local_dir}")

try:
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False  # 确保下载实际文件而不是软链接
    )
    print("\n下载完成！")
    print(f"请重新运行 benchmark.py: python benchmark.py")
except Exception as e:
    print(f"\n下载失败: {e}")
    print("请检查网络连接或 Hugging Face 访问权限。")
