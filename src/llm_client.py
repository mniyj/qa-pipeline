"""
llm_client.py
统一的 LLM 调用客户端，支持 Anthropic / DashScope(通义千问) / OpenAI

特性：
- 统一接口：call(prompt, system) → str
- 自动重试 + 指数退避
- 速率控制
- 成本跟踪
"""

import os
import json
import time
import asyncio
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


class _RateLimitError(Exception):
    """429 限流错误，携带服务端建议的等待时间"""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limited (Retry-After={retry_after}s)")


@dataclass
class UsageStats:
    """API 使用量统计"""
    total_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    failed_calls: int = 0
    total_cost_yuan: float = 0.0

    # 各模型的大致单价（元/千token），可按需调整
    PRICING = {
        # Anthropic
        "claude-sonnet-4-20250514": {"input": 0.021, "output": 0.105},
        # DashScope (通义千问)
        "qwen-plus": {"input": 0.004, "output": 0.012},
        "qwen-max": {"input": 0.02, "output": 0.06},
        "qwen-turbo": {"input": 0.002, "output": 0.006},
        # OpenAI
        "gpt-4o": {"input": 0.018, "output": 0.072},
        "gpt-4o-mini": {"input": 0.001, "output": 0.004},
        # DeepSeek
        "deepseek-chat": {"input": 0.002, "output": 0.008},
        "deepseek-reasoner": {"input": 0.004, "output": 0.016},
    }

    def track(self, model: str, input_tokens: int, output_tokens: int):
        self.total_calls += 1
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        pricing = self.PRICING.get(model, {"input": 0.01, "output": 0.03})
        cost = (
            input_tokens / 1000 * pricing["input"]
            + output_tokens / 1000 * pricing["output"]
        )
        self.total_cost_yuan += cost

    def summary(self) -> str:
        return (
            f"调用统计: {self.total_calls}次 | "
            f"输入 {self.total_input_tokens:,} tokens | "
            f"输出 {self.total_output_tokens:,} tokens | "
            f"失败 {self.failed_calls}次 | "
            f"估算成本 ¥{self.total_cost_yuan:.2f}"
        )


# 全局统计
usage_stats = UsageStats()


class LLMClient:
    """统一的 LLM 调用客户端"""

    def __init__(
        self,
        provider: str,
        model: str,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        max_retries: int = 3,
        requests_per_minute: int = 30,
    ):
        self.provider = provider
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_retries = max_retries
        self.min_interval = 60.0 / requests_per_minute
        self._last_call_time = 0.0

    def _rate_limit(self):
        """简单的速率控制"""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.min_interval:
            time.sleep(self.min_interval - elapsed)
        self._last_call_time = time.time()

    def call(
        self,
        prompt: str,
        system: str = "",
        response_format: str = "text",  # "text" or "json"
    ) -> str:
        """
        统一调用接口
        返回模型输出的文本
        """
        for attempt in range(self.max_retries):
            try:
                self._rate_limit()

                if self.provider == "anthropic":
                    result = self._call_anthropic(prompt, system)
                elif self.provider == "dashscope":
                    result = self._call_dashscope(prompt, system)
                elif self.provider == "openai":
                    result = self._call_openai(prompt, system)
                elif self.provider == "deepseek":
                    result = self._call_deepseek(prompt, system)
                else:
                    raise ValueError(f"不支持的 provider: {self.provider}")

                return result

            except Exception as e:
                usage_stats.failed_calls += 1
                wait = (2 ** attempt) * 2  # 2, 4, 8 秒
                logger.warning(
                    f"调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}. "
                    f"等待 {wait}s 后重试..."
                )
                if attempt < self.max_retries - 1:
                    time.sleep(wait)
                else:
                    logger.error(f"调用彻底失败: {e}")
                    raise

    def _call_anthropic(self, prompt: str, system: str) -> str:
        """调用 Anthropic Claude API"""
        import anthropic

        client = anthropic.Anthropic(
            api_key=os.environ.get("ANTHROPIC_API_KEY")
        )

        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system

        response = client.messages.create(**kwargs)

        # 跟踪 token 使用量
        usage_stats.track(
            self.model,
            response.usage.input_tokens,
            response.usage.output_tokens,
        )

        return response.content[0].text

    def _call_dashscope(self, prompt: str, system: str) -> str:
        """
        调用通义千问 (DashScope) API
        文档：https://help.aliyun.com/zh/model-studio/
        """
        import openai as openai_module

        # DashScope 兼容 OpenAI SDK
        client = openai_module.OpenAI(
            api_key=os.environ.get("DASHSCOPE_API_KEY"),
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        # 跟踪用量
        if response.usage:
            usage_stats.track(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        return response.choices[0].message.content

    def _call_deepseek(self, prompt: str, system: str) -> str:
        """调用 DeepSeek API（兼容 OpenAI SDK）"""
        import openai as openai_module

        client = openai_module.OpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if response.usage:
            usage_stats.track(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        return response.choices[0].message.content

    def _call_openai(self, prompt: str, system: str) -> str:
        """调用 OpenAI API"""
        import openai as openai_module

        client = openai_module.OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY")
        )

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=self.max_tokens,
            temperature=self.temperature,
        )

        if response.usage:
            usage_stats.track(
                self.model,
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
            )

        return response.choices[0].message.content

    def call_batch(
        self,
        prompts: list[str],
        system: str = "",
        concurrency: int = 3,
    ) -> list[str]:
        """批量调用（串行，带速率控制）"""
        results = []
        for i, prompt in enumerate(prompts):
            logger.info(f"  批量调用 {i+1}/{len(prompts)}")
            try:
                result = self.call(prompt, system)
                results.append(result)
            except Exception as e:
                logger.error(f"  批量调用失败 #{i+1}: {e}")
                results.append("")
        return results

    async def acall(self, prompt: str, system: str = "") -> str:
        """异步调用（用于并发场景）"""
        _BASE_URLS = {
            "deepseek": "https://api.deepseek.com",
            "dashscope": "https://dashscope.aliyuncs.com/compatible-mode/v1",
            "openai": "https://api.openai.com/v1",
        }
        _API_KEYS = {
            "deepseek": os.environ.get("DEEPSEEK_API_KEY"),
            "dashscope": os.environ.get("DASHSCOPE_API_KEY"),
            "openai": os.environ.get("OPENAI_API_KEY"),
        }

        for attempt in range(self.max_retries):
            try:
                if self.provider in _BASE_URLS:
                    return await self._acall_openai_compat(
                        prompt, system,
                        _BASE_URLS[self.provider],
                        _API_KEYS[self.provider],
                    )
                elif self.provider == "anthropic":
                    return await self._acall_anthropic(prompt, system)
                else:
                    raise ValueError(f"不支持的 provider: {self.provider}")
            except _RateLimitError as e:
                # 429 限流：优先使用 Retry-After，否则指数退避（最少 10s）
                usage_stats.failed_calls += 1
                wait = e.retry_after if e.retry_after else max(10, (2 ** attempt) * 5)
                logger.warning(
                    f"触发限流 429 (尝试 {attempt+1}/{self.max_retries})，"
                    f"等待 {wait}s 后重试..."
                )
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    raise
            except Exception as e:
                usage_stats.failed_calls += 1
                wait = (2 ** attempt) * 2
                logger.warning(f"async 调用失败 (尝试 {attempt+1}/{self.max_retries}): {e}. 等待 {wait}s...")
                if attempt < self.max_retries - 1:
                    await asyncio.sleep(wait)
                else:
                    raise

    async def _acall_openai_compat(
        self, prompt: str, system: str, base_url: str, api_key: str
    ) -> str:
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        async with httpx.AsyncClient(timeout=120.0) as client:
            response = await client.post(
                f"{base_url}/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": self.model,
                    "messages": messages,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                },
            )
            if response.status_code == 429:
                # 尊重服务端返回的 Retry-After，而不是盲目使用指数退避
                retry_after = int(response.headers.get("Retry-After", 0)) or None
                raise _RateLimitError(retry_after=retry_after)
            response.raise_for_status()
            data = response.json()

        usage = data.get("usage", {})
        usage_stats.track(
            self.model,
            usage.get("prompt_tokens", 0),
            usage.get("completion_tokens", 0),
        )
        return data["choices"][0]["message"]["content"]

    async def _acall_anthropic(self, prompt: str, system: str) -> str:
        import anthropic

        client = anthropic.AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        kwargs = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if system:
            kwargs["system"] = system
        response = await client.messages.create(**kwargs)
        usage_stats.track(self.model, response.usage.input_tokens, response.usage.output_tokens)
        return response.content[0].text


def create_client(config: dict, stage: str) -> LLMClient:
    """
    根据配置创建 LLM 客户端
    stage: seed_generation / expansion / quality_check
    """
    model_config = config.get("models", {}).get(stage, {})
    return LLMClient(
        provider=model_config.get("provider", "dashscope"),
        model=model_config.get("model", "qwen-plus"),
        max_tokens=model_config.get("max_tokens", 4096),
        temperature=model_config.get("temperature", 0.7),
    )
