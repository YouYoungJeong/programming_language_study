"""
Ollama 도구로 Local LLM model 사용
    - https://ollama.com/download 설치
    - pip install ollama
"""

from ollama import chat
question = '여름철 장마에 대해 설명해줘'

# ollama의 chat양식
response = chat(
    # model = "exaone3.5:latest",
    model = "gemma:latest",
    messages = [
        {
            'role' : 'user',
            'content' : question
        }
    ]
)
print('답변 :',response.message.content)