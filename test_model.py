# import 
from llama_cpp import Llama


model_name = 'llama-7b.ggmlv3.q4_0.bin'
# Load your llm
llm = Llama(model_path = f'./model/{model_name}')

# pass prompt to the model 
response = llm('what is deep learning?')

print('hi')
# check response 
print(response['choices'][0]['text'])