from typing import List, Union, Tuple
import numpy as np
from registrable import Registrable

class Metrics(Registrable):
    @staticmethod
    def calculate(**kwargs):
        return NotImplementedError

@Metrics.register('accuracy')
class Accuracy(Metrics):
    @staticmethod
    def calculate(prediction, truth, force_lower=True) -> float:
        if force_lower:
            prediction = prediction.lower()
            truth = truth.lower()
        return 1.0 if prediction == truth else 0.0

@Metrics.register('hit_ratio')
class HitRatio(Metrics):
    @staticmethod
    def calculate(retrieved_int: List[int], truth: List[int], hit_num=3) -> float:
        truth = truth if isinstance(truth, list) else [truth]
        hit = len(set(truth).intersection(set(retrieved_int[:hit_num])))
        hit_rate = hit / len(truth)
        return hit_rate

@Metrics.register('rouge_l')
class ROUGE(Metrics):
    @staticmethod
    def calculate(generation: str, truth: str) -> float:
        try:
            from rouge_score import rouge_scorer
        except ImportError:
            raise ImportError("Please install rouge_score: pip install rouge_score")
        
        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = scorer.score(generation, truth)
        return scores["rougeL"].fmeasure

@Metrics.register('cosine_similarity')
class CosSim(Metrics):
    @staticmethod
    def calculate(generated_text: str, reference_texts: Union[str, List[str]], model_name):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError("Please install sentence_transformers: pip install sentence-transformers")

        # Load/Reference model
        model = CosSim._load_sentence_transformer_model(model_name)
        generated_embedding = model.encode([generated_text])[0]

        if isinstance(reference_texts, str):
            reference_embedding = model.encode([reference_texts])[0]
            similarity_score = np.dot(generated_embedding, reference_embedding) / (
                    np.linalg.norm(generated_embedding) * np.linalg.norm(reference_embedding))
            return max(similarity_score, 0)
        else:
            similarity_scores = []
            for reference_text in reference_texts:
                reference_embedding = model.encode([reference_text])[0]
                individual_score = np.dot(generated_embedding, reference_embedding) / (
                        np.linalg.norm(generated_embedding) * np.linalg.norm(reference_embedding))
                similarity_scores.append(individual_score)
            return max(np.mean(similarity_scores), 0)

    @staticmethod
    def _load_sentence_transformer_model(model_name: str):
        """Cache for sentence transformer models"""
        if not hasattr(CosSim, '_model_cache'):
            CosSim._model_cache = {}
        
        if model_name not in CosSim._model_cache:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
            except ImportError:
                raise ImportError("Please install required packages: pip install sentence-transformers torch")
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = SentenceTransformer(model_name).to(device)
            CosSim._model_cache[model_name] = model
        
        return CosSim._model_cache[model_name]

@Metrics.register('bleu')
class BLEU(Metrics):
    @staticmethod
    def calculate(generated_text: str, reference_text: str, is_japanese: bool = False) -> float:
        try:
            import sacrebleu
        except ImportError:
            raise ImportError("Please install sacrebleu: pip install sacrebleu")

        generated_text = generated_text.lstrip("\n").rstrip("\n").split("\n")[0]
        reference = [reference_text.strip()]

        if is_japanese:
            bleu = sacrebleu.corpus_bleu([generated_text], [[ref] for ref in reference], 
                                       tokenize='ja-mecab', lowercase=True)
        else:
            bleu = sacrebleu.corpus_bleu([generated_text], [[ref] for ref in reference], lowercase=True)

        return bleu.score / 100

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

@Metrics.register('micro_f1')
class MicroF1(Metrics):
    @staticmethod
    def calculate(extracted_entities: List[str], ground_truth_entities: List[str]) -> Tuple[int, int, int]:
        """
        Calculates true positives, false positives, and false negatives for entity extraction.

        This function compares a list of extracted entities against a list of ground truth entities
        to determine the count of true positives (correctly extracted entities), false positives
        (incorrectly extracted entities), and false negatives (missed entities).

        Both lists are case-insensitive, and leading/trailing spaces in extracted entities are ignored.

        Parameters:
        - extracted_entities (List[str]): The list of entities extracted by the model.
        - ground_truth_entities (List[str]): The list of actual entities (ground truth).

        Returns:
        - Tuple[int, int, int]: A tuple containing the counts of true positives, false positives, and false negatives.
        """
        # Normalize the extracted entities by making them lowercase and stripping leading/trailing spaces
        normalized_extracted_entities = [entity.lower().strip() for entity in extracted_entities]

        # Normalize the ground truth entities by making them lowercase
        normalized_ground_truth_entities = [entity.lower() for entity in ground_truth_entities]

        # Calculate true positives by finding the intersection between extracted and ground truth entities
        true_positives = len(set(normalized_extracted_entities).intersection(set(normalized_ground_truth_entities)))

        # Calculate false positives as extracted entities not in ground truth
        false_positives = len(normalized_extracted_entities) - true_positives

        # Calculate false negatives as ground truth entities not extracted
        false_negatives = len(normalized_ground_truth_entities) - true_positives

        return true_positives, false_positives, false_negatives

@Metrics.register('ndcg')
class NDCG(Metrics):
    @staticmethod
    def calculate(predicted_relevance_scores: List[int], true_relevance_weights: List[float]) -> float:
        """
        Calculates and evaluates the Normalized Discounted Cumulative Gain (NDCG) score directly from predicted relevance scores
        against true relevance weights. It normalizes the scores to ensure a fair comparison, trimming the predicted scores
        if necessary to match the length of the true relevance weights.

        Parameters:
        - predicted_relevance_scores (List[int]): Indices of items ranked by the algorithm, expected to be integers starting from 1.
        - true_relevance_weights (List[float]): Actual relevance weights for the items, with higher values indicating greater relevance.

        Returns:
        - float: The NDCG score, normalized against the ideal ranking, ranging from 0 to 1.
        """
        # Trim the predicted scores to match the true scores length if necessary
        if len(predicted_relevance_scores) > len(true_relevance_weights):
            predicted_relevance_scores = predicted_relevance_scores[:len(true_relevance_weights)]

        dcg, idcg = 0.0, 0.0

        # Calculate DCG for the predicted ranking
        for i, score_index in enumerate(predicted_relevance_scores, start=1):
            if score_index - 1 < len(true_relevance_weights):
                relevance = true_relevance_weights[score_index - 1]
            else:
                relevance = 0
            dcg += (np.power(2, relevance) - 1) / np.log2(i + 1)

        # Calculate IDCG using sorted true relevance weights
        for i, weight in enumerate(sorted(true_relevance_weights, reverse=True), start=1):
            idcg += (np.power(2, weight) - 1) / np.log2(i + 1)

        # Avoid division by zero
        return 0 if idcg == 0 else dcg / idcg
