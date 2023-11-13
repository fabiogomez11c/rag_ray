import warnings
from templates import base_template, base_human_template
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

warnings.filterwarnings("ignore")


class LLM:
    """LLM object designed for our endpoint"""

    def generate_code(self, user_request: str, generated_context: str) -> str:
        """Generate the desired code based on the user input"""
        chat_model = ChatOpenAI()
        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", base_template), ("human", base_human_template)]
        )
        chain = LLMChain(prompt=chat_prompt, llm=chat_model)
        result = chain.run()
        return result


if __name__ == "__main__":
    llm = LLM()
    res = llm.generate_code(
        user_html="<button>Click me!</button>",
        user_request="I want this button to have light colors and a hover effect, make it whole circle button, away from the standards even if you need to make it bigger to be a circle.",
    )
    print(res)
