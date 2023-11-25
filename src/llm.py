import os

from dotenv import load_dotenv
from google.cloud import aiplatform
from vertexai.preview.language_models import TextGenerationModel

load_dotenv()


def load_llm_model(
    model_name: str = "text-bison",
    llm_location: str = "us-central1",
    project_id: str = os.environ.get("PROJECT_ID"),
) -> TextGenerationModel:
    """This function loads the Vertex AI model.

    Parameters:
    model_name (str): The name of the Vertex AI model.

    Returns:
    Vertex AI model: The Vertex AI model.
    """
    aiplatform.init(project=project_id, location=llm_location)
    model = TextGenerationModel.from_pretrained(model_name)
    # see Doc: https://github.com/googleapis/python-aiplatform/blob/main/vertexai/language_models/_language_models.py
    return model


def online_predict_with_llm_model(
    input_text: str,
    prompt_format: str,
    llm_model: TextGenerationModel = load_llm_model(),
    max_output_tokens: int = 1024,
    temperature: float = 0.1,
    top_p: float = 0.8,
    top_k: int = 40,
):
    """
    This function uses Vertex AI for text prediction. It receives text and model parameters
    and returns the sentiment and catégorie of the text as a response from the Vertex model.
    One-shot prompt
    Parameters:
    text (str): The input text for which sentiment needs to be predicted.

    Returns:
    str: The predicted sentiment of the input text.
    """
    response = llm_model.predict(
        prompt=prompt_format.format(text=input_text),
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        top_k=top_k,
    )
    return response.text.strip()
