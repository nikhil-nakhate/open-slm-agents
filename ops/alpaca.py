"""Alpaca instruction format for GPT-2 and other instruction-tuned models.

The Alpaca format is a standard for instruction-tuning:
- Wraps user input in instruction formatting
- Supports optional input fields
- Prompts for structured responses

Reference: https://github.com/tatsu-lab/stanford_alpaca
"""

from typing import Optional


def format_alpaca_prompt(
    instruction: str,
    input_text: Optional[str] = None,
    system_prompt: Optional[str] = None
) -> str:
    """Format a prompt in Alpaca instruction format.

    Args:
        instruction: The user's instruction/question
        input_text: Optional additional input/context
        system_prompt: Optional system-level instructions (prepended)

    Returns:
        Formatted prompt string ready for model input

    Examples:
        >>> format_alpaca_prompt("What is 2+2?")
        'Below is an instruction...\\n\\n### Instruction:\\nWhat is 2+2?\\n\\n### Response:\\n'

        >>> format_alpaca_prompt("Summarize this", "The quick brown fox...")
        'Below is an instruction...\\n\\n### Instruction:\\nSummarize this\\n\\n### Input:\\nThe quick brown fox...\\n\\n### Response:\\n'
    """
    parts = []

    # Add system prompt if provided
    if system_prompt:
        parts.append(system_prompt.strip())
        parts.append("")  # Blank line after system prompt

    # Standard Alpaca prefix
    parts.append(
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
    )
    parts.append("")  # Blank line

    # Instruction
    parts.append(f"### Instruction:\n{instruction.strip()}")
    parts.append("")  # Blank line

    # Optional input field
    if input_text and input_text.strip():
        parts.append(f"### Input:\n{input_text.strip()}")
        parts.append("")  # Blank line

    # Response marker (model completes from here)
    parts.append("### Response:")

    return "\n".join(parts)


def extract_alpaca_response(generated_text: str) -> str:
    """Extract just the response portion from generated Alpaca-formatted text.

    Args:
        generated_text: Full generated text (including prompt)

    Returns:
        Just the response portion after ### Response:
    """
    if "### Response:" in generated_text:
        parts = generated_text.split("### Response:", 1)
        if len(parts) > 1:
            return parts[1].strip()

    # Fallback: return everything after the instruction/input
    if "### Instruction:" in generated_text:
        # Try to find where the actual generation starts
        parts = generated_text.split("### Instruction:")
        if len(parts) > 1:
            # Look for the end of the instruction section
            rest = parts[1]
            if "\n\n" in rest:
                # Skip past instruction and input sections
                sections = rest.split("\n\n")
                # Last section should be the response
                return sections[-1].strip()

    return generated_text.strip()
