import warnings
from templates import base_template, base_human_template
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

warnings.filterwarnings("ignore")


class LLM:
    """LLM object designed for our endpoint"""

    def __init__(self, conversation_phase: str):
        self.conversation_phase = conversation_phase

    def generate_code(self, user_html: str, user_request: str) -> str:
        """Generate the desired code based on the user input"""
        chat_model = ChatOpenAI()
        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", base_template), ("human", base_human_template)]
        )
        chain = LLMChain(prompt=chat_prompt, llm=chat_model)
        result = chain.run({"html": user_html, "request": user_request})
        return result


if __name__ == "__main__":
    llm = LLM(conversation_phase="none")
    res = llm.generate_code(
        user_html="<button>Click me!</button>",
        user_request="I want this button to have light colors and a hover effect, make it whole circle button, away from the standards even if you need to make it bigger to be a circle.",
    )
    print(res)
