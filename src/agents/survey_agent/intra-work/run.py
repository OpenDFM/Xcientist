import subprocess
import os

def run_docker_command(container_id, script_path, args=None):
    """
    在指定Docker容器中运行Python脚本
    """
    cmd = ["docker", "exec", container_id, "python3", script_path]
    if args:
        cmd.extend(args)
    
    try:
        result = subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"命令执行失败: {' '.join(cmd)}")
        return 0
    return 1

# 主程序
TASK_ID = "semantic_scholar_test_pipeline"
QUERY = "AI automatic overview/survey generation"
BASE_DIR = "/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work/"

returncode = 1

returncode = run_docker_command(
    "9133f8e337c7",
    "/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work/scripts/retriever.py",
    ["-p", BASE_DIR, "-t", TASK_ID, "-q", QUERY, "-m", "5"]
)

print("RETRIEVING over, converting to md...")
# 运行第二个命令
if returncode == 1:  # 只有第一个命令成功才执行第二个
    returncode = run_docker_command(
        "c5babbd6ff8d",
        "/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work/scripts/convert_pdf_to_md.py",
        ["-p", BASE_DIR, "-t", TASK_ID]
    )

print("CONVERTING over, EXTRACTING...")
if returncode == 1:
    returncode = run_docker_command(
        "9133f8e337c7",
        "/hpc_stor03/sjtu_home/ziyue.yang/sci-agent/deep-survey/intra-work/scripts/extraction.py",
        ["-p", BASE_DIR, "-t", TASK_ID]
    )