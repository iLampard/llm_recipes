from transformers import Trainer
from easyllm_kit.utils.log_utils import default_logger as logger


# Debugging: Print the evaluation metrics after training
def print_evaluation_metrics(trainer: Trainer):
    eval_result = trainer.evaluate()
    logger.info("Evaluation Metrics:", eval_result)


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
    logger.info(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def print_trainable_layers(model):
    # print trainable parameters for inspection
    logger.info("Trainable parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"\t{name}")
    return
