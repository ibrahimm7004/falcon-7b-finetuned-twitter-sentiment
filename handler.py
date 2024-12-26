from typing import Dict, Any, List
from transformers import AutoModelForCausalLM, AutoTokenizer


class EndpointHandler:
    def __init__(self, path=""):
        """
        Initializes the EndpointHandler with the fine-tuned Falcon model and tokenizer.

        Args:
            path (str): The Hugging Face model repo or local path containing the model files.
        """
        # Load the fine-tuned model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True)

        # Default generation parameters
        self.generation_config = {
            "max_new_tokens": 1,  # Limit length to prevent verbose responses
            "num_return_sequences": 1,  # Return only one response
            "repetition_penalty": 2.0,  # Penalize repeated tokens
            "temperature": 0.7,  # Control randomness
            "top_k": 2,  # Limit vocabulary to top 2 likely tokens
            "top_p": 0.95,  # Use nucleus sampling to focus on high-probability tokens
            "pad_token_id": self.tokenizer.eos_token_id,  # Ensure padding uses EOS token
            "eos_token_id": self.tokenizer.eos_token_id,  # End generation at EOS token
        }

    def __call__(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handles inference requests and generates predictions.

        Args:
            data (Dict[str, Any]): Input payload with the key "inputs" for the input text.

        Returns:
            Dict[str, Any]: A dictionary containing the sentiment output.
        """
        # Extract the input prompt
        input_text = data.get("inputs", "")

        # Format input for the model
        formatted_input = f"### Human: {input_text}\n### Assistant:"

        # Tokenize the input
        inputs = self.tokenizer(formatted_input, return_tensors="pt")
        inputs.pop("token_type_ids", None)  # Remove 'token_type_ids' if it exists

        # Generate the response
        outputs = self.model.generate(
            **inputs,
            **self.generation_config
        )

        # Decode and extract the sentiment
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        sentiment = response.split("### Assistant:")[1].strip()  # Extract sentiment

        return {"sentiment": sentiment}
