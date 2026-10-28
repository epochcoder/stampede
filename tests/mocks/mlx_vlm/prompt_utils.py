def apply_chat_template(processor, config, prompt, add_generation_prompt=True, return_messages=False, num_images=0, num_audios=0, **kw):
    assert num_images == 1 and config == {"model_type": "qwen3_vl"}
    return "<chat>" + prompt
