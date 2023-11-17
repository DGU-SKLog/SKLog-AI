from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai

app = FastAPI()
from pathlib import Path
import environ
import os
#uvicorn main:app --reload
# Build paths inside the project like this: BASE_DIR / 'subdir'.

BASE_DIR = Path(__file__).resolve().parent

environ.Env.read_env(
    env_file=os.path.join(BASE_DIR, '.env')
)

env = environ.Env(
    API_KEY=(str, ''),
)
API_KEY = env.str('API_KEY', default='')

openai.api_key = API_KEY

model = "gpt-3.5-turbo"

class InputText(BaseModel):
    request: str
    content: str
    userContentList: Optional[List[str]] = None
    aiContentList: Optional[List[str]] = None

@app.post("/query")
def text(input_text: InputText):

    query = f"""
{input_text.request}
    
내용: 
{input_text.content}

 """

    if input_text.userContentList:
        query += "*참고: 아래 예시 참고해서 대답해.\n"

        for user_item, ai_item in zip(input_text.userContentList, input_text.aiContentList):
            query += f"""
변경전 예시:
{user_item}
            
변경후 예시:
{ai_item}
            
            """

    print("query: ", query)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        temperature=0,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']

    print("answer: ", answer)
    return {'content': answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)