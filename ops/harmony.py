"""Harmony structured response format for GPT-OSS models.

The Harmony format is OpenAI's structured format for GPT-OSS models, supporting:
- Multi-turn conversations
- Chain-of-thought reasoning
- Tool calls and responses
- Multiple output channels (final, reasoning, browser, python, etc.)

Reference: https://github.com/openai/harmony
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Union

from pydantic import BaseModel, Field, field_validator


class MessageRole(str, Enum):
    """Valid roles for Harmony messages."""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


class AssistantChannel(str, Enum):
    """Valid channels for assistant responses in Harmony format."""
    FINAL = "final"  # Final user-facing response
    REASONING = "reasoning"  # Internal chain-of-thought
    BROWSER = "browser"  # Web browsing tool use
    PYTHON = "python"  # Python code execution
    CALL = "call"  # Function/tool calls
    CONSTRAIN = "constrain"  # Constrained generation


class HarmonyMessage(BaseModel):
    """A single message in the Harmony format.

    Examples:
        User message:
        <|start|>user<|message|>What is 2+2?<|end|>

        Assistant with reasoning:
        <|start|>assistant<|channel|>reasoning<|message|>Let me think...<|end|>
        <|start|>assistant<|channel|>final<|message|>The answer is 4<|end|>
    """
    role: MessageRole
    content: str
    channel: Optional[AssistantChannel] = None  # Only for assistant messages

    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v: Optional[AssistantChannel], info) -> Optional[AssistantChannel]:
        """Ensure channel is only set for assistant messages."""
        if v is not None and info.data.get('role') != MessageRole.ASSISTANT:
            raise ValueError(f"channel can only be set for assistant messages, got role={info.data.get('role')}")
        return v

    def to_harmony_string(self) -> str:
        """Convert message to Harmony format string.

        Returns:
            Formatted string like: <|start|>user<|message|>content<|end|>
        """
        parts = [f"<|start|>{self.role.value}"]

        if self.channel:
            parts.append(f"<|channel|>{self.channel.value}")

        parts.append(f"<|message|>{self.content}<|end|>")

        return "".join(parts)

    @classmethod
    def from_harmony_string(cls, text: str) -> HarmonyMessage:
        """Parse a Harmony format string into a HarmonyMessage.

        Args:
            text: Harmony formatted string

        Returns:
            Parsed HarmonyMessage

        Raises:
            ValueError: If the string is not valid Harmony format
        """
        if not text.startswith("<|start|>"):
            raise ValueError("Harmony message must start with <|start|>")
        if not text.endswith("<|end|>"):
            raise ValueError("Harmony message must end with <|end|>")

        # Remove start and end tags
        inner = text[len("<|start|>"):-len("<|end|>")]

        # Parse role
        if "<|message|>" not in inner:
            raise ValueError("Harmony message must contain <|message|>")

        parts = inner.split("<|message|>", 1)
        header = parts[0]
        content = parts[1]

        # Extract role and optional channel
        channel = None
        if "<|channel|>" in header:
            role_part, channel_part = header.split("<|channel|>", 1)
            role = MessageRole(role_part)
            channel = AssistantChannel(channel_part)
        else:
            role = MessageRole(header)

        return cls(role=role, content=content, channel=channel)


class HarmonyConversation(BaseModel):
    """A multi-turn conversation in Harmony format.

    Examples:
        Simple Q&A:
        >>> conv = HarmonyConversation()
        >>> conv.add_user_message("What is 2+2?")
        >>> conv.add_assistant_message("4", channel=AssistantChannel.FINAL)
        >>> print(conv.to_harmony_string())

        With reasoning:
        >>> conv = HarmonyConversation()
        >>> conv.add_user_message("Solve x^2 = 16")
        >>> conv.add_assistant_message("Let me think step by step...", channel=AssistantChannel.REASONING)
        >>> conv.add_assistant_message("x = ±4", channel=AssistantChannel.FINAL)
    """
    messages: List[HarmonyMessage] = Field(default_factory=list)
    system_prompt: Optional[str] = None

    def add_message(self, message: HarmonyMessage) -> None:
        """Add a message to the conversation."""
        self.messages.append(message)

    def add_user_message(self, content: str) -> None:
        """Add a user message to the conversation."""
        self.messages.append(HarmonyMessage(role=MessageRole.USER, content=content))

    def add_assistant_message(
        self,
        content: str,
        channel: AssistantChannel = AssistantChannel.FINAL
    ) -> None:
        """Add an assistant message to the conversation.

        Args:
            content: Message content
            channel: Output channel (default: FINAL for user-facing response)
        """
        self.messages.append(HarmonyMessage(
            role=MessageRole.ASSISTANT,
            content=content,
            channel=channel
        ))

    def add_system_message(self, content: str) -> None:
        """Add a system message to the conversation."""
        self.messages.append(HarmonyMessage(role=MessageRole.SYSTEM, content=content))

    def to_harmony_string(self) -> str:
        """Convert entire conversation to Harmony format string.

        Returns:
            Full conversation in Harmony format, ready for model input
        """
        parts = []

        # Add system prompt if present
        if self.system_prompt:
            parts.append(HarmonyMessage(
                role=MessageRole.SYSTEM,
                content=self.system_prompt
            ).to_harmony_string())

        # Add all messages
        for msg in self.messages:
            parts.append(msg.to_harmony_string())

        return "".join(parts)

    def to_prompt_string(self, include_assistant_prefix: bool = True) -> str:
        """Convert conversation to a prompt string for generation.

        Args:
            include_assistant_prefix: If True, adds assistant prefix for generation

        Returns:
            Formatted prompt ready for model input
        """
        base = self.to_harmony_string()

        if include_assistant_prefix:
            # Add assistant channel prefix to prompt completion
            base += "<|start|>assistant<|channel|>final<|message|>"

        return base

    @classmethod
    def from_harmony_string(cls, text: str) -> HarmonyConversation:
        """Parse a full Harmony conversation from string.

        Args:
            text: Full Harmony formatted conversation

        Returns:
            Parsed HarmonyConversation
        """
        messages = []

        # Split on <|start|> to get individual messages
        parts = text.split("<|start|>")

        for part in parts:
            if not part.strip():
                continue

            # Reconstruct the message with <|start|> prefix
            msg_text = "<|start|>" + part

            # Ensure it ends with <|end|>
            if not msg_text.endswith("<|end|>"):
                # Find the next <|end|>
                if "<|end|>" in msg_text:
                    msg_text = msg_text[:msg_text.index("<|end|>") + len("<|end|>")]
                else:
                    continue

            try:
                msg = HarmonyMessage.from_harmony_string(msg_text)
                messages.append(msg)
            except ValueError:
                # Skip invalid messages
                continue

        return cls(messages=messages)

    def get_final_response(self) -> Optional[str]:
        """Extract the final user-facing response from the conversation.

        Returns:
            Content of the last assistant message with channel=final, or None
        """
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT and msg.channel == AssistantChannel.FINAL:
                return msg.content
        return None

    def get_reasoning_chain(self) -> List[str]:
        """Extract all reasoning messages (chain-of-thought).

        Returns:
            List of reasoning message contents in order
        """
        return [
            msg.content
            for msg in self.messages
            if msg.role == MessageRole.ASSISTANT and msg.channel == AssistantChannel.REASONING
        ]


class HarmonyParser:
    """Utility for parsing Harmony format responses from model output."""

    @staticmethod
    def parse_generated_response(generated_text: str, prompt: str) -> HarmonyConversation:
        """Parse model output into a structured conversation.

        Args:
            generated_text: Full text generated by the model (including prompt)
            prompt: Original prompt text to strip from the beginning

        Returns:
            Parsed conversation with all messages
        """
        # Remove the original prompt to get just the generated part
        if generated_text.startswith(prompt):
            response_only = generated_text[len(prompt):]
        else:
            response_only = generated_text

        # Parse the full conversation (prompt + response)
        return HarmonyConversation.from_harmony_string(generated_text)

    @staticmethod
    def extract_final_answer(generated_text: str) -> Optional[str]:
        """Extract just the final user-facing answer from generated text.

        Args:
            generated_text: Full generated text

        Returns:
            Final answer content, or None if not found
        """
        conv = HarmonyConversation.from_harmony_string(generated_text)
        return conv.get_final_response()

    @staticmethod
    def stream_parse(token_stream: str) -> Optional[HarmonyMessage]:
        """Parse a streaming token buffer to extract complete messages.

        Args:
            token_stream: Accumulated tokens from streaming generation

        Returns:
            Parsed message if complete, None if incomplete
        """
        if not token_stream.endswith("<|end|>"):
            return None  # Message not complete yet

        # Find the last complete message
        if "<|start|>" not in token_stream:
            return None

        # Get the last message
        last_start = token_stream.rfind("<|start|>")
        last_msg = token_stream[last_start:]

        try:
            return HarmonyMessage.from_harmony_string(last_msg)
        except ValueError:
            return None


# Convenience functions for common use cases

def create_simple_prompt(user_message: str, system_prompt: Optional[str] = None) -> str:
    """Create a simple Harmony-formatted prompt for a single user message.

    Args:
        user_message: The user's question or request
        system_prompt: Optional system instructions

    Returns:
        Harmony-formatted prompt ready for generation
    """
    conv = HarmonyConversation(system_prompt=system_prompt)
    conv.add_user_message(user_message)
    return conv.to_prompt_string()


def parse_response(generated_text: str) -> Dict[str, Any]:
    """Parse a generated response and extract key information.

    Args:
        generated_text: Full generated text from the model

    Returns:
        Dictionary with 'final_answer', 'reasoning_chain', and 'full_conversation'
    """
    conv = HarmonyConversation.from_harmony_string(generated_text)

    return {
        "final_answer": conv.get_final_response(),
        "reasoning_chain": conv.get_reasoning_chain(),
        "full_conversation": conv,
        "all_messages": conv.messages,
    }
