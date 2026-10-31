"""HuggingFace 访问配置 —— 在 import datasets/transformers 之前调用。

国内网络访问不了 huggingface.co 时,把下载端点切到镜像站。
用法(在脚本最顶部,任何 from datasets/transformers 之前):

    from src.hf_config import setup_hf_endpoint
    setup_hf_endpoint()          # 默认用 hf-mirror.com

    # 或显式指定:
    setup_hf_endpoint("https://hf-mirror.com")

    # 或关闭(回到官方 huggingface.co):
    setup_hf_endpoint(None)

已设过 HF_ENDPOINT 环境变量时尊重你的设置,不覆盖。
"""
import os
from typing import Optional

DEFAULT_MIRROR = "https://hf-mirror.com"


def setup_hf_endpoint(endpoint: Optional[str] = "__unset__") -> Optional[str]:
    """设置 HF_ENDPOINT。返回最终生效的端点。

    endpoint=None           -> 清除,用官方
    endpoint="__unset__"    -> 沿用已有环境变量,没有就用默认镜像
    endpoint="<url>"        -> 强制用该 URL
    """
    current = os.environ.get("HF_ENDPOINT")

    if endpoint == "__unset__":
        if current:
            return current                      # 尊重用户已有设置
        os.environ["HF_ENDPOINT"] = DEFAULT_MIRROR
        return DEFAULT_MIRROR

    if endpoint is None:
        os.environ.pop("HF_ENDPOINT", None)     # 清除,走官方
        return None

    os.environ["HF_ENDPOINT"] = endpoint
    return endpoint


def report() -> None:
    """打印当前生效的端点,便于排查。"""
    ep = os.environ.get("HF_ENDPOINT") or "https://huggingface.co (官方)"
    print(f"[hf] HF_ENDPOINT = {ep}")
