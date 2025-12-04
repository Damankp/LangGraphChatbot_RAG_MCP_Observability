from langchain_openai import ChatOpenAI
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint, HuggingFaceEmbeddings, HuggingFaceEndpointEmbeddings
from dotenv import load_dotenv

load_dotenv()


# model1 = ChatHuggingFace(llm = HuggingFaceEndpoint(
#     repo_id="openai/gpt-oss-120b",
#     task="text-generation"))

# model2 = ChatHuggingFace(llm = HuggingFaceEndpoint(
#     repo_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
#     task="text-generation"))

# model3 = ChatOpenAI(model='gpt-4')

# print(model3.invoke('Hi bot, what is your name').content)


embedding = HuggingFaceEndpointEmbeddings(repo_id='sentence-transformers/all-MiniLM-L6-v2',
                                  task='feature-extraction')

# model = ChatHuggingFace(llm=embedding)

documents = [
    "Delhi is the capital of India",
    "Kolkata is the capital of West Bengal",
    "Paris is the capital of France"
]

vector = embedding.embed_documents(documents)

print(str(vector))