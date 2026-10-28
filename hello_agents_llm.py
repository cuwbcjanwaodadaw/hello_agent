"""Hello Agents 的 OpenAI 兼容流式 LLM 客户端。"""

import os
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI, OpenAIError


class HelloAgentsLLM:
    """优先使用显式参数，否则读取项目 .env 或系统环境变量。"""

    def __init__(
        self,
        model: Optional[str] = None,
        apiKey: Optional[str] = None,
        baseUrl: Optional[str] = None,
        timeout: Optional[float] = None,
    ):
        # 不依赖启动命令所在目录；已设置的系统环境变量优先。
        load_dotenv(Path(__file__).resolve().parent / ".env")
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout if timeout is not None else float(os.getenv("LLM_TIMEOUT", "60"))

        settings = {
            "LLM_MODEL_ID": self.model,
            "LLM_API_KEY": apiKey,
            "LLM_BASE_URL": baseUrl,
        }
        placeholders = {"YOUR-MODEL", "YOUR-API-KEY", "YOUR-URL"}
        missing = [name for name, value in settings.items()
                   if not value or value.strip() in placeholders]
        if missing:
            raise ValueError("请在 .env 中填写有效配置：" + ", ".join(missing))
        if timeout <= 0:
            raise ValueError("timeout / LLM_TIMEOUT 必须大于 0。")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
    ) -> Optional[str]:
        """流式打印并返回最终回答；API 调用失败时返回 None。

        默认省略 temperature，使用服务默认值。MiniMax M3 系列固定使用 1。
        此基础封装只返回正文，后续工具调用需要另行保留完整消息。
        """
        print(f"正在调用 {self.model} 模型...", flush=True)
        options = {}
        if temperature is not None:
            options["temperature"] = temperature

        try:
            collected_content = []
            with self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                stream=True,
                **options,
            ) as response:
                for chunk in response:
                    # 用量统计分片可能没有 choices；推理分片可能没有正文。
                    if not chunk.choices:
                        continue
                    content = chunk.choices[0].delta.content
                    if content:
                        print(content, end="", flush=True)
                        collected_content.append(content)
            print()
            return "".join(collected_content)
        except OpenAIError as exc:
            print(f"\n调用 LLM API 时发生错误：{exc}")
            return None

    def close(self) -> None:
        """释放客户端连接。"""
        self.client.close()


def main() -> int:
    try:
        llm_client = HelloAgentsLLM()
    except ValueError as exc:
        print(exc)
        return 1

    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant that writes Python code."},
            {"role": "user", "content": "写一个快速排序算法"},
        ]
        response = llm_client.think(messages)
        return 0 if response else 1
    finally:
        llm_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
