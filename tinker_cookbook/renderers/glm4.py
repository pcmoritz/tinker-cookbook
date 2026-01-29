"""
GLM-4.7 family renderers.

Includes:
- GLM4Renderer: Base GLM-4.7 with thinking enabled
- GLM4DisableThinkingRenderer: GLM-4.7 with thinking disabled

Reference: https://huggingface.co/zai-org/GLM-4.7-Flash
"""

import json

import tinker

from tinker_cookbook.renderers.base import (
    Message,
    RenderContext,
    RenderedMessage,
    Renderer,
    ToolCall,
    ToolSpec,
    UnparsedToolCall,
    _tool_call_payload,
    ensure_text,
    parse_content_blocks,
    parse_response_for_stop_token,
    remove_thinking,
)
from tinker_cookbook.tokenizer_utils import Tokenizer


class GLM4Renderer(Renderer):
    """
    Renderer for GLM-4.7 models with thinking enabled.

    This renderer is designed to match HuggingFace's GLM-4.7 chat template behavior.
    The format uses [gMASK]<sop> prefix followed by role tokens.

    Reference: https://huggingface.co/zai-org/GLM-4.7-Flash

    Format:
        [gMASK]<sop><|system|>
        You are a helpful assistant.<|user|>
        What can you help me with?<|assistant|>
        <think>
        [reasoning content]
        </think>
        I can help you with...<|endoftext|>

    The default strip_thinking_from_history=True matches HF behavior where thinking
    blocks are stripped from historical assistant messages in multi-turn conversations.
    Use strip_thinking_from_history=False for multi-turn RL to get the extension property.
    """

    def __init__(self, tokenizer: Tokenizer, strip_thinking_from_history: bool = True):
        """
        Args:
            tokenizer: The tokenizer to use for encoding.
            strip_thinking_from_history: When True (default), strips <think>...</think> blocks
                from assistant messages in multi-turn history. This matches HuggingFace's
                GLM-4.7 chat template behavior. Set to False to preserve thinking in history
                (useful for multi-turn RL where you need the extension property).
        """
        super().__init__(tokenizer)
        self.strip_thinking_from_history = strip_thinking_from_history

    @property
    def has_extension_property(self) -> bool:
        """Extension property depends on strip_thinking_from_history setting.

        When strip_thinking_from_history=False, thinking blocks are preserved in
        history, so each successive observation is a prefix extension of the previous.

        When strip_thinking_from_history=True (default), thinking blocks are stripped
        from historical messages, breaking the extension property.
        """
        return not self.strip_thinking_from_history

    @property
    def _bos_tokens(self) -> list[int]:
        """Return BOS tokens: [gMASK]<sop>"""
        return self.tokenizer.encode("[gMASK]<sop>", add_special_tokens=False)

    def _get_role_token(self, role: str) -> str:
        """Get the role token string for a given role."""
        if role == "tool":
            return "<|observation|>"
        return f"<|{role}|>"

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        role = message["role"]
        role_token = self._get_role_token(role)
        header_str = f"{role_token}\n"

        content = message["content"]

        if isinstance(content, list):
            # Structured content - handle with list operations
            parts = content
            if (
                self.strip_thinking_from_history
                and message["role"] == "assistant"
                and not ctx.is_last
            ):
                # Remove thinking parts for historical messages
                parts = remove_thinking(parts)
            # Render parts in order, preserving interleaved thinking/text structure.
            rendered_parts = []
            for p in parts:
                if p["type"] == "thinking":
                    rendered_parts.append(f"<think>{p['thinking']}</think>")
                elif p["type"] == "text":
                    rendered_parts.append(p["text"])
                # ToolCallPart handled via message's tool_calls field
            output_content = "".join(rendered_parts)
        else:
            # String content - pass through as-is.
            output_content = content

        # Handle tool_calls field
        if "tool_calls" in message:
            output_content += "\n" + "\n".join(
                [
                    f"<tool_call>\n{json.dumps(_tool_call_payload(tool_call))}\n</tool_call>"
                    for tool_call in message["tool_calls"]
                ]
            )

        # Add end token for non-empty content
        if output_content:
            output_content += "<|endoftext|>"

        header = tinker.types.EncodedTextChunk(
            tokens=self.tokenizer.encode(header_str, add_special_tokens=False)
        )
        output: list[tinker.ModelInputChunk] = [
            tinker.types.EncodedTextChunk(
                tokens=self.tokenizer.encode(output_content, add_special_tokens=False)
            )
        ]
        return RenderedMessage(header=header, output=output)

    @property
    def _end_message_token(self) -> int:
        tokens = self.tokenizer.encode("<|endoftext|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|endoftext|>, got {len(tokens)}"
        return tokens[0]

    def _get_user_token(self) -> int:
        """Get the <|user|> token ID for stop sequence."""
        tokens = self.tokenizer.encode("<|user|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|user|>, got {len(tokens)}"
        return tokens[0]

    def _get_observation_token(self) -> int:
        """Get the <|observation|> token ID for stop sequence."""
        tokens = self.tokenizer.encode("<|observation|>", add_special_tokens=False)
        assert len(tokens) == 1, f"Expected single token for <|observation|>, got {len(tokens)}"
        return tokens[0]

    def get_stop_sequences(self) -> list[int]:
        """Return stop sequences: <|endoftext|>, <|user|>, <|observation|>"""
        return [self._end_message_token, self._get_user_token(), self._get_observation_token()]

    def _get_generation_suffix(self, role: str, ctx: RenderContext) -> list[int]:
        """Return tokens to append to the prompt for generation.

        For assistant role, adds <|assistant|>\\n<think> to trigger thinking mode.
        """
        role_token = self._get_role_token(role)
        suffix_str = f"{role_token}\n"
        if role == "assistant":
            suffix_str += "<think>"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            # Also check for <|user|> and <|observation|> as stop tokens
            user_token = self._get_user_token()
            obs_token = self._get_observation_token()
            for stop_token in [user_token, obs_token]:
                if stop_token in response:
                    str_response = self.tokenizer.decode(response[: response.index(stop_token)])
                    assistant_message = Message(role="assistant", content=str_response)
                    parse_success = True
                    break

        if not parse_success:
            return assistant_message, False

        # Prepend <think> since we prefill with it in generation
        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]
        if not content.startswith("<think>") and "</think>" in content:
            content = "<think>" + content

        # Parse <think>...</think> and <tool_call>...</tool_call> blocks together
        parts = parse_content_blocks(content)

        if parts is not None:
            assistant_message["content"] = parts

            # Also populate tool_calls and unparsed_tool_calls fields for backward compatibility
            tool_calls = [p["tool_call"] for p in parts if p["type"] == "tool_call"]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            unparsed = [
                UnparsedToolCall(raw_text=p["raw_text"], error=p["error"])
                for p in parts
                if p["type"] == "unparsed_tool_call"
            ]
            if unparsed:
                assistant_message["unparsed_tool_calls"] = unparsed
        else:
            assistant_message["content"] = content

        return assistant_message, True

    def to_openai_message(self, message: Message) -> dict:
        """Convert a Message to OpenAI API format with reasoning_content for thinking.

        GLM-4.7's API accepts either:
        - message['reasoning_content'] as a separate field
        - <think>...</think> embedded in content

        We use reasoning_content for cleaner separation.
        """
        result: dict = {"role": message["role"]}

        content = message["content"]
        if isinstance(content, str):
            result["content"] = content
        else:
            # Extract thinking into reasoning_content, keep text in content
            thinking_parts = []
            text_parts = []
            for p in content:
                if p["type"] == "thinking":
                    thinking_parts.append(p["thinking"])
                elif p["type"] == "text":
                    text_parts.append(p["text"])
                # Skip tool_call/unparsed_tool_call - handled via tool_calls field

            result["content"] = "".join(text_parts)
            if thinking_parts:
                result["reasoning_content"] = "".join(thinking_parts)

        # Handle tool_calls
        if "tool_calls" in message and message["tool_calls"]:
            result["tool_calls"] = [
                {
                    "type": "function",
                    "id": tc.id,
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    },
                }
                for tc in message["tool_calls"]
            ]

        # Handle tool response fields
        if message["role"] == "tool":
            if "tool_call_id" in message:
                result["tool_call_id"] = message["tool_call_id"]
            if "name" in message:
                result["name"] = message["name"]

        return result

    def create_conversation_prefix_with_tools(
        self, tools: list[ToolSpec], system_prompt: str = ""
    ) -> list[Message]:
        """Create system message with GLM-4.7 tool specifications.

        GLM-4.7 uses XML `<tools>` tags containing JSON tool definitions in OpenAI format,
        appended to the system message content (similar to Qwen3).
        """
        tools_text = ""
        if tools:
            tool_lines = "\n".join(
                json.dumps({"type": "function", "function": tool}, separators=(", ", ": "))
                for tool in tools
            )
            tools_text = f"""# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_lines}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""

        # Add separator between system prompt and tools if system prompt exists
        if system_prompt:
            content = system_prompt + "\n\n" + tools_text
        else:
            content = tools_text

        return [Message(role="system", content=content)]


class GLM4DisableThinkingRenderer(GLM4Renderer):
    """
    Renderer for GLM-4.7 models with thinking disabled.

    This renderer matches HuggingFace's GLM-4.7 chat template behavior with
    enable_thinking=False. It does not add the <think> prefill and strips
    any thinking blocks from content.

    Use this renderer when you want to train or sample from GLM-4.7 models in
    "non-thinking" mode.
    """

    @property
    def has_extension_property(self) -> bool:
        """Non-thinking mode always satisfies extension - no thinking to strip from history."""
        return True

    def _get_generation_suffix(self, role: str, ctx: RenderContext) -> list[int]:
        """Return tokens to append to the prompt for generation.

        For non-thinking mode, does NOT add <think> prefill.
        """
        role_token = self._get_role_token(role)
        suffix_str = f"{role_token}\n"
        return self.tokenizer.encode(suffix_str, add_special_tokens=False)

    def render_message(self, message: Message, ctx: RenderContext) -> RenderedMessage:
        """Render message in non-thinking mode.

        For assistant messages, strips any ThinkingPart from structured content.
        """
        if message["role"] == "assistant":
            content = message["content"]

            # Strip thinking from content
            if isinstance(content, list):
                # Remove ThinkingPart, keep only text
                text_content = "".join(p["text"] for p in content if p["type"] == "text")
                message = message.copy()
                message["content"] = text_content

        return super().render_message(message, ctx)

    def parse_response(self, response: list[int]) -> tuple[Message, bool]:
        """Parse response without expecting thinking blocks."""
        assistant_message, parse_success = parse_response_for_stop_token(
            response, self.tokenizer, self._end_message_token
        )
        if not parse_success:
            # Also check for <|user|> and <|observation|> as stop tokens
            user_token = self._get_user_token()
            obs_token = self._get_observation_token()
            for stop_token in [user_token, obs_token]:
                if stop_token in response:
                    str_response = self.tokenizer.decode(response[: response.index(stop_token)])
                    assistant_message = Message(role="assistant", content=str_response)
                    parse_success = True
                    break

        if not parse_success:
            return assistant_message, False

        # Parse <tool_call>...</tool_call> blocks (no thinking expected)
        assert isinstance(assistant_message["content"], str)
        content = assistant_message["content"]
        parts = parse_content_blocks(content)

        if parts is not None:
            assistant_message["content"] = parts

            tool_calls = [p["tool_call"] for p in parts if p["type"] == "tool_call"]
            if tool_calls:
                assistant_message["tool_calls"] = tool_calls

            unparsed = [
                UnparsedToolCall(raw_text=p["raw_text"], error=p["error"])
                for p in parts
                if p["type"] == "unparsed_tool_call"
            ]
            if unparsed:
                assistant_message["unparsed_tool_calls"] = unparsed
        else:
            assistant_message["content"] = content

        return assistant_message, True
