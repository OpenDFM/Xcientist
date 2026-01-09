import sys
import os

# 将项目根目录加入路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openai import OpenAI
from src.agents.experiment_agent.shared.utils.config import (
    MINIMAX_API_KEY,
    MINIMAX_API_BASE,
    MINIMAX_MODEL_EXTRA_BODY,
)


def test_minimax_connection():
    """
    测试MiniMax-M2 API是否可连通
    返回: True表示连通，False表示不连通
    """
    try:
        print("正在测试MiniMax-M2连通性...")

        # 创建OpenAI客户端（使用MiniMax配置）
        client = OpenAI(api_key=MINIMAX_API_KEY, base_url=MINIMAX_API_BASE)

        # 发送一个简单的测试请求
        response = client.chat.completions.create(
            model="MiniMax-M2",
            messages=[{"role": "user", "content": "测试连接，请回复OK"}],
            max_tokens=10,
            extra_body=MINIMAX_MODEL_EXTRA_BODY,
        )

        # 如果能获得响应，说明连通
        if response and response.choices:
            print("✓ MiniMax-M2连通性测试成功！")
            return True
        else:
            print("✗ MiniMax-M2响应异常")
            return False

    except Exception as e:
        print(f"✗ MiniMax-M2连通性测试失败: {str(e)}")
        return False


def main():
    """主函数"""
    is_connected = test_minimax_connection()

    sys.exit(0 if is_connected else 1)


if __name__ == "__main__":
    main()
