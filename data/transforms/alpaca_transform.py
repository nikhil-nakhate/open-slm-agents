from typing import Dict, Tuple, Callable


def format_input(entry: Dict) -> str:
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry.get('instruction', '')}"
    )
    input_field = entry.get("input", "")
    input_text = f"\n\n### Input:\n{input_field}" if input_field else ""
    return instruction_text + input_text


class AlpacaTransform:
    """Formats instruction-tuning entries into (prompt, target) text pairs.

    Input entry is expected to contain keys: instruction, input (optional), output.
    Matches the notebook implementation with proper response formatting.
    """

    def __call__(self, entry: Dict) -> Tuple[str, str]:
        prompt = format_input(entry)
        # Add the response formatting as shown in the notebook
        response_text = f"\n\n### Response:\n{entry.get('output', '')}"
        target = response_text
        return prompt, target


def build_alpaca_transform() -> Callable[[Dict], Tuple[str, str]]:
    return AlpacaTransform()

