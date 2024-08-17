from transformers import Trainer

# Debugging: Print the evaluation metrics after training
def print_evaluation_metrics(trainer: Trainer):
    eval_result = trainer.evaluate()
    print("Evaluation Metrics:", eval_result)

