from langchain import PromptTemplate
from langchain.llms import LlamaCpp
from langchain.chains import LLMChain

# Load llm 

llm = LlamaCpp(model_path = './model/llama-7b.ggmlv3.q4_0.bin')
# Define a template 

template = 'Tell me a breif detail about {something}'

prompt = PromptTemplate.from_template(template)

# print(prompt.input_variables)

# formatted_prompt = prompt.format(something = 'deep learning')

# print(llm(prompt = formatted_prompt, llm = llm, stop = ['Q:', '\n']))


# define chain 

llm_chain = LLMChain(prompt=prompt, llm = llm)

llm_chain.run("Deep Learning")