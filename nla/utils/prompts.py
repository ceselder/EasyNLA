"""AV prompt construction shared by the RL trainers (and eval/probe scripts)."""


def build_prompt_text(prompt_msgs, inject_char, tokenizer):
    """Apply the chat template; substitute the <INJECT> placeholder with inject_char."""
    msgs = [
        {**m, "content": m["content"].replace("<INJECT>", inject_char)}
        if isinstance(m.get("content"), str)
        else m
        for m in prompt_msgs
    ]
    # enable_thinking=False: hybrid-reasoning templates (Qwen3/3.5/3.6) otherwise
    # open a bare `<think>\n` block in the generation prompt — the AV would be
    # trained/rolled-out inside an unclosed think block. False renders a closed
    # empty block, so explanations start at the normal answer position. Must
    # match schema.compute_canonical_neighbors + train_sft's template calls.
    return tokenizer.apply_chat_template(
        msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
