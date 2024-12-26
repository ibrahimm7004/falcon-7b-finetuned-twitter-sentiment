from transformers import AutoModelForCausalLM, AutoTokenizer


class FalconModel:
    def __init__(self, model_name="ibrahim7004/falcon-7b-finetuned-twitter"):
        """
        Initializes the model and tokenizer.
        This ensures the model is loaded only once when the FalconModel class is instantiated.
        """
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, trust_remote_code=True)

    def predict_sentiment(self, input_text):
        formatted_input = f"### Human: {input_text}\n### Assistant:"

        # Tokenize input
        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        # Remove 'token_type_ids' if it exists
        inputs.pop("token_type_ids", None)

        # Generate output
        outputs = self.model.generate(
            **inputs,
            max_new_tokens=1,
            num_return_sequences=1,
            repetition_penalty=2.0,
            temperature=0.7,
            top_k=2,
            top_p=0.95,
        )

        # Decode and extract response
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = response.split("### Assistant:")[1].strip()
        return sentiment


# Create a reusable instance of the model (to load only once)
falcon_model_instance = FalconModel()
