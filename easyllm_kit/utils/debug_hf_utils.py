from transformers import Trainer


# Debugging: Print the evaluation metrics after training
def print_evaluation_metrics(trainer: Trainer):
    eval_result = trainer.evaluate()
    message = f"Evaluation Metrics: {eval_result}"
    return message


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    message = f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    return message


def print_trainable_layers(model):
    # print trainable parameters for inspection
    message = "Trainable layers:\n"
    for name, param in model.named_parameters():
        if param.requires_grad:
            message += f"\t{name}\n"
    return message.strip()  # Remove trailing newline
    
