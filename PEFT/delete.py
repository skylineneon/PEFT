import os
import shutil
import time


def delete_first_folder_if_more_than_three(folder_path):
    # 检查文件夹路径是否存在
    if not os.path.exists(folder_path):
        print("指定的文件夹不存在")
        return

    # 获取文件夹中的所有文件夹并按名称排序
    folders = [item for item in os.listdir(
        folder_path) if os.path.isdir(os.path.join(folder_path, item))]

    # 如果文件夹中的子文件夹数量不大于3
    if len(folders) <= 3:
        print("子文件夹数量不大于3，不执行删除操作")
        return
    folders.sort()  # 对文件夹列表进行排序
    # 获取第一个文件夹的完整路径
    first_folder = os.path.join(folder_path, folders[0])

    # 删除第一个文件夹
    try:
        shutil.rmtree(first_folder)
        print(f"已删除文件夹：{first_folder}")
    except Exception as e:
        print(f"删除文件夹时出错：{e}")


# 指定要操作的文件夹路径
folder_path = '/root/workspace/PEFT/checkpoints/lora/qwen'

while True:
    delete_first_folder_if_more_than_three(folder_path)
    # 等待300秒（5分钟）
    time.sleep(2)
