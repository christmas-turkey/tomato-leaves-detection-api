from typing import List
from dotenv import load_dotenv
from langchain_core.prompts.prompt import PromptTemplate
from langchain.chains.llm import LLMChain
from langchain_fireworks import Fireworks

# Load API keys from .env file
load_dotenv()


class ModelNotAvailableError(Exception):
    pass


AVAILIBLE_MODELS = [
    "llama-v3p1-405b-instruct",
    "llama-v3p1-70b-instruct",
    "llama-v3p1-8b-instruct",
    "llama-v3-70b-instruct",
    "mixtral-8x22b-instruct",
    "mixtral-8x7b-instruct"
]

PROMPT_TEMPLATE = """
You are an expert on tomato leaf diseases and you are purposed to help farmers identify the diseases on their tomato plants.
You will be given with the list of tomato leaves diseases and your task is to provide the description and treatment (or possible preventions) for each of the provided diseases.
If the list of provided diseases is empty, just say "No diseases detected".

The detected diseases list: {detected_diseases}
Your answer:
"""

PROMPT = PromptTemplate(
    template=PROMPT_TEMPLATE,
    input_variables=["detected_diseases"]
)

class LLMAssistant:
    """ This class generates descriptions of tomato leaf diseases based on the provided list of diseases. """

    def __init__(
        self, 
        model_name: str="llama-v3p1-405b-instruct", 
        temperature: int=0, 
        max_tokens: int=3000
    ):
        """
        Initialize the LLMAssistant class.

        Args:
            model_name: The name of the LLM to use.
            temperature: The temperature of the model.
            max_tokens: The maximum number of tokens to generate.
        """

        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.llm = self._initialize_llm(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        self.llm_chain = LLMChain(
            llm=self.llm,
            prompt=PROMPT
        )
    
    def _initialize_llm(
        self,
        model_name: str="llama-v3p1-405b-instruct", 
        temperature: int=0, 
        max_tokens: int=3000
    ) -> Fireworks:
        """
        Initialize a FireworksAI LLM instance with the provided parameters.

        Args:
            model_name: The name of the LLM to use.
            temperature: The temperature of the model.
            max_tokens: The maximum number of tokens to generate.
        
        Returns:
            Fireworks LLM instance
        """

        if model_name not in AVAILIBLE_MODELS:
            raise ModelNotAvailableError(f"Model name {model_name} is not available. Please choose from {AVAILIBLE_MODELS}")

        llm = Fireworks(
            model=f"accounts/fireworks/models/{model_name}",
            temperature=temperature,
            max_tokens=max_tokens
        )

        return llm

    def generate_response(
        self, 
        diseases_labels: List[str],
        model_name: str="llama-v3p1-405b-instruct",
    ) -> str:
        """
        Generate a response for the provided list of diseases.

        Args:
            diseases_labels: The list of diseases to generate a response for.
            model_name: The name of the LLM to use.
        
        Returns:
            The generated response.
        """

        # Reinitalize the model if a different model name is provided
        if model_name != self.model_name:
            self.llm = self._initialize_llm(
                model_name=model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )
            
            self.llm_chain = LLMChain(
                llm=self.llm,
                prompt=PROMPT
            )

            self.model_name = model_name
        
        response = self.llm_chain.run(diseases_labels)

        return response
        
