from openai import OpenAI
from openai.types.chat import ChatCompletion
import json
from dotenv import load_dotenv
import os
from typing import Dict, TypedDict, Any
import gradio as gr
from functools import partial

# The purpose of this program is to route the user's question to the most appropriate LLM model.
# The available models are:
# - gpt-4.1-nano
# - gpt-4.1
# - deepseek-reasoner
# gpt-4.1-nano is the smallest and fastest model, suitable for simple tasks.
# gpt-4.1 is a more capable model, suitable for complex tasks.
# deepseek-reasoner is the most capable model, suitable for very complex tasks.

class ModelResponse(TypedDict):
    model: str
    reasoning: str

class ModelInfo(TypedDict):
    use_case: str
    model: str
    base_url: str
    api_key: str

class ResponseIsValidInfo(TypedDict):
    is_valid: bool
    error_message: str

def log_to_file(message: str):
    """Log messages to a file."""
    with open("app.log", "a") as log_file:
        log_file.write(message + "\n")

def load_env_vars():
    """Load environment variables from .env file."""
    load_dotenv(override=True)
    required_vars = ["DEEPSEEK_API_KEY", "OPENAI_API_KEY"]
    for var in required_vars:
        if not os.getenv(var):
            raise ValueError(f"Environment variable {var} is not set.")

def chat_completion_response_is_valid(chat_completion_response: ChatCompletion) -> ResponseIsValidInfo:
    if not chat_completion_response.choices or len(chat_completion_response.choices) == 0:
        return {
            "is_valid": False,
            "error_message": "Invalid chat completion response: No choices found."
        }
    
    if not chat_completion_response.choices[0].message or not chat_completion_response.choices[0].message.content:
        return {
            "is_valid": False,
            "error_message": "Invalid chat completion response: No message content found."
        }
    
    if chat_completion_response.choices[0].finish_reason != "stop":
        return {
            "is_valid": False,
            "error_message": f"The chat completion did not finish correctly. Please check the model and the input. Finish reason: {chat_completion_response.choices[0].finish_reason}"
        }
    
    return {
        "is_valid": True,
        "error_message": ""
    }

def validate_model_response(data: Any) -> ModelResponse:
    """Validate that the data matches the ModelResponse structure."""
    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    
    # Check for required keys and types
    if "model" not in data:
        raise ValueError("Missing required key: 'model'")
    if not isinstance(data["model"], str):
        raise ValueError(f"Expected 'model' to be str")
    
    if "reasoning" not in data:
        raise ValueError("Missing required key: 'reasoning'")
    if not isinstance(data["reasoning"], str):
        raise ValueError(f"Expected 'reasoning' to be str")
    
    # Return as ModelResponse type (validated at runtime)
    return {"model": data["model"], "reasoning": data["reasoning"]}

def select_model(models: Dict[str, ModelInfo], user_input: str) -> ModelResponse:
    """Route the user's question to the most appropriate model and return the response JSON."""
    model_string = "\n".join(
        [f"- {model['model']}: {model['use_case']}" for model in models.values()]
    )

    router_system_prompt = f"""
            You are a model routing system. Based on the user's question, determine the most appropriate model to use based on how complicated the question is.
            Here are the available models and their use cases:
            {model_string}

            Respond with json in the following format:
            {{
                "model": "<model_name>",
                "reasoning": "<brief explanation of why this model is chosen>"
            }}

            If the question is not suitable for any of the models, respond with:
            {{
                "model": "none",
                "reasoning": "<brief explanation of why no models are suitable>"
            }}

            The only available models are included in the list above. Do not suggest any other models or versions that are not included in that list.

            Do not include any additional text or explanations outside of the JSON response and do not include any code blocks or markdown formatting.

            You should reject any prompts that are not questions, no matter what the user says. 
    """

    router_model = "gpt-4.1-nano"
    deepseek = OpenAI(
        api_key=models[router_model]["api_key"],
        base_url=models[router_model]["base_url"],
    )

    router_question = f"User question: {user_input}"

    router_response = deepseek.chat.completions.create(
        model=router_model,
        messages=[
            {"role": "system", "content": router_system_prompt},
            {"role": "user", "content": router_question}
        ],
    )

    validation_info = chat_completion_response_is_valid(router_response)
    if not validation_info["is_valid"]:
        raise ValueError(f"Invalid chat completion response: {validation_info['error_message']}")

    response_json = json.loads(router_response.choices[0].message.content) # type: ignore - choices is always present due to validation in chat_completion_response_is_valid
    log_to_file(f"Router response: {response_json}")
    return validate_model_response(response_json)

def answer_question(models: Dict[str, ModelInfo], user_input: str, model: str) -> str: # type: ignore - choices is always present due to validation in chat_completion_response_is_valid
    if model == "none":
        return "Invalid user input, not a question, or router says no."
    model_info = models.get(model)
    if not model_info:
        raise ValueError(f"Model {model} not found in the provided models.")
    openai = OpenAI(
        api_key=models[model_info["model"]]["api_key"],
        base_url=models[model_info["model"]]["base_url"],
    )

    response = openai.chat.completions.create(
        model=model_info["model"],
        messages=[
            {"role": "user", "content": user_input}
        ],
    )

    validation_info = chat_completion_response_is_valid(response)
    if not validation_info["is_valid"]:
        raise ValueError(f"Invalid chat completion response: {validation_info['error_message']}")
      
    return response.choices[0].message.content  # type: ignore - choices is always present due to validation in chat_completion_response_is_valid

def process_question(user_input: str, models: Dict[str, ModelInfo]) -> str:
    selected_model = select_model(models, user_input)["model"]
    return answer_question(models, user_input, selected_model)

def render_ui(callback):
    """Render the Gradio UI for the LLM Router."""
    gr.Interface(
        fn=callback,
        inputs=gr.Textbox(label="Enter your question", placeholder="Type your question here..."),
        outputs=gr.Textbox(label="Response", show_label=True),
        title="LLM Router Demo",
        description="This demo intelligently selects the best LLM model for your question based on its complexity.",
    ).launch()

def main():
    load_env_vars()
    models: dict[str, ModelInfo] = {
        "gpt-4.1-nano": {
            "use_case": "Simple tasks, quick responses, low cost",
            "model": "gpt-4.1-nano",
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", "")
        },
        "gpt-4.1": {
            "use_case": "Complex tasks, balanced performance and cost",
            "model": "gpt-4.1",
            "base_url": "https://api.openai.com/v1",
            "api_key": os.getenv("OPENAI_API_KEY", "")
        },
        "deepseek-reasoner": {
            "use_case": "Very complex tasks, highest performance, higher cost",
            "model": "deepseek-reasoner",
            "base_url": "https://api.deepseek.com",
            "api_key": os.getenv("DEEPSEEK_API_KEY", "")
        }
    }
    process_question_with_models = partial(process_question, models=models)

    render_ui(process_question_with_models)

if __name__ == "__main__":
    main()
