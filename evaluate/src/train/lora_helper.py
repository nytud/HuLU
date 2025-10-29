from peft import LoraConfig, TaskType, get_peft_model

from train.arguments import Arguments


def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def choose_lora_target(model):
    choose_target = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "query",
        "key",
        "value",
        "q",
        "v",
        "k",
        "o",
        "query_key_value",
        "dense",
        "dense_h_to_4h",
        "dense_4h_to_h",
    ]
    target_set = set()

    for name, _ in model.named_modules():
        layer_name = name.split(".")[-1]
        if layer_name in choose_target:
            target_set.add(layer_name)

    return list(target_set)


def set_lora(hulu_args: Arguments, sequente_classification=False, model=None):
    target_modules = choose_lora_target(model)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS if sequente_classification else None,
        r=hulu_args.lora_r,
        lora_alpha=hulu_args.lora_alpha,
        lora_dropout=hulu_args.lora_dropout,
        target_modules=target_modules,
    )
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    return model
