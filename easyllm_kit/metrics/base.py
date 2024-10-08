from typing import List, Union, Tuple

import evaluate
import numpy as np
import rouge_scorer

from registrable import Registrable
from sentence_transformers import SentenceTransformer
import torch


class Metrics(Registrable):
    pass


@Metrics.register('accuracy')
class Accuracy(Metrics):
    @staticmethod
    def calculate(prediction, truth) -> bool:
        return prediction == truth


@Metrics.register('hit_ratio')
class HitRatio(Metrics):
    @staticmethod
    def calculate(retrieved_int: List[int], truth: List[int], hit_num=3) -> float:
        # in case truth is one integer value
        truth = truth if isinstance(truth, list) else [truth]
        # Calculate the number of hits within the top 3 retrieved integers
        hit = len(set(truth).intersection(set(retrieved_int[:hit_num])))
        # Normalize the hit count by the total number of truth integers to get the hit rate
        hit_rate = hit / len(truth)
        return hit_rate


@Metrics.register('rouge_l')
class ROUGE(Metrics):
    @staticmethod
    def calculate(generation: str, truth: str) -> float:
        # Initialize the ROUGE scorer with the ROUGE-L metric
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        # Calculate the ROUGE scores between the generated text and the truth text
        scores = scorer.score(generation, truth)
        # Extract and return the ROUGE-L F-measure score
        return scores["rougeL"].fmeasure


def load_sentence_transformer_model(model_name: str) -> SentenceTransformer:
    """
    Loads a Sentence Transformer model by its name and moves it to the appropriate device.

    Parameters:
    - model_name (str): The name of the model to load.

    Returns:
    - SentenceTransformer: The loaded SentenceTransformer model.
    """

    global sentence_transformer_model_cache

    # a model cache ensure we do not load the model on every call
    if model_name not in sentence_transformer_model_cache:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name).to(device)
        sentence_transformer_model_cache[model_name] = model

    return sentence_transformer_model_cache[model_name]


@Metrics.register('cosine_similarity')
class CosSim(Metrics):
    @staticmethod
    def calculate(generated_text: str, reference_texts: Union[str, List[str]],
                  model_name):
        # Load/Reference model
        model = load_sentence_transformer_model(model_name)

        # Embedding for the generated text
        generated_embedding = model.encode([generated_text])[0]

        # Handling a single reference text
        if isinstance(reference_texts, str):
            # Embedding for the single reference text
            reference_embedding = model.encode([reference_texts])[0]
            # Compute cosine similarity
            similarity_score = np.dot(generated_embedding, reference_embedding) / (
                    np.linalg.norm(generated_embedding) * np.linalg.norm(reference_embedding))
            # Ensure non-negative score
            return max(similarity_score, 0)

        # Handling multiple reference texts
        else:
            similarity_scores = []
            for reference_text in reference_texts:
                # Embedding for each reference text
                reference_embedding = self.model.encode([reference_text])[0]
                # Compute cosine similarity for each reference
                individual_score = np.dot(generated_embedding, reference_embedding) / (
                        np.linalg.norm(generated_embedding) * np.linalg.norm(reference_embedding))
                similarity_scores.append(individual_score)
            # Calculate and ensure non-negative average score
            return max(np.mean(similarity_scores), 0)

@Metrics.register('bleu')
class BLEU(Metrics):
    @staticmethod
    def calculate(generated_text: str, reference_text: str, is_japanese: bool = False) -> float:
        """
        Calculates the BLEU score for a generated text compared to a reference truth text. This function supports
        both general text and Japanese-specific evaluation by using the sacrebleu library.

        Parameters:
        - generated_text (str): The generated text to be evaluated.
        - reference_text (str): The reference truth text.
        - is_japanese (bool, optional): Flag to indicate whether the text is in Japanese, requiring special tokenization.

        Returns:
        - float: The BLEU score as a percentage (0 to 1 scale) for the generated text against the reference truth.
        """
        global sacrebleu
        if sacrebleu is None:
            sacrebleu = evaluate.load("sacrebleu")

        # Preprocess input texts
        generated_text = generated_text.lstrip("\n").rstrip("\n").split("\n")[0]
        candidate = [generated_text]
        reference = [[reference_text]]

        # Compute BLEU score with or without Japanese-specific tokenization
        bleu_args = {"predictions": candidate, "references": reference, "lowercase": True}
        if is_japanese:
            bleu_args["tokenize"] = "ja-mecab"
        score = sacrebleu.compute(**bleu_args)["score"] / 100

        return score

@Metrics.register('f1')
class F1(Metrics):
    @staticmethod
    def calculate(metrics_list: List[Tuple[int, int, int]]) -> float:
        """
        Calculates the F1 score from a list of tuples containing true positives, false positives, and false negatives.

        Parameters:
        - metrics_list (List[Tuple[int, int, int]]): A list of tuples, where each tuple contains counts of true positives,
          false positives, and false negatives in that order for various classifications or entity extractions.

        Returns:
        - float: The computed F1 score, ranging from 0 to 1.
        """
        total_tp, total_fp, total_fn = 0, 0, 0

        # Aggregate total true positives, false positives, and false negatives
        for tp, fp, fn in metrics_list:
            total_tp += tp
            total_fp += fp
            total_fn += fn

        # Calculate precision and recall
        precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0

        # Calculate F1 score, handling the case where precision + recall equals 0
        if precision + recall == 0:
            return 0
        else:
            return 2 * precision * recall / (precision + recall)