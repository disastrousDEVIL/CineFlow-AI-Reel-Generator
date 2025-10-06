from langchain_google_vertexai import ChatVertexAI

# Initialize Gemini via LangChain
llm = ChatVertexAI(
    model="gemini-2.5-flash",    # or "gemini-1.5-pro"
    project="reel-automation-using-vertex",
    location="us-central1",
    temperature=0.7,
    # max_output_tokens=512,
)

# Basic conversation
response = llm.invoke("Write 5 cinematic scenes about a boy looking at the stars.")
print(response.content)
