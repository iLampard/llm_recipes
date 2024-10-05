from easyllm_kit.models.base import LLM

@LLM.register("mixtral")
class MixtralLLM(LLM):
    def load(self, **kwargs):
        model_path = kwargs.get('model_path', 'mistralai/Mixtral-8x7B-v0.1')
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to(self.device)

    def generate(self, prompt: str, **kwargs) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=kwargs.get('max_tokens', 100),
            temperature=kwargs.get('temperature', 0.7),
            top_p=kwargs.get('top_p', 0.9),
            repetition_penalty=kwargs.get('repetition_penalty', 1.1)
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
