import warnings
from templates import base_template, base_human_template
from langchain.chat_models import ChatOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from langchain.chains import LLMChain

warnings.filterwarnings("ignore")


class LLM:
    """LLM object designed for our endpoint"""

    def get_response(self, user_request: str, generated_context: str = "") -> str:
        """Generate the desired code based on the user input"""
        chat_model = ChatOpenAI(temperature=0)
        chat_prompt = ChatPromptTemplate.from_messages(
            [("system", base_template), ("human", base_human_template)]
        )
        chain = LLMChain(prompt=chat_prompt, llm=chat_model)
        result = chain.run({"request": user_request, "context": generated_context})
        return result


if __name__ == "__main__":
    llm = LLM()
    res = llm.generate_code(
        user_request="How to install ray in python?", generated_context=""
    )
    print(res)
