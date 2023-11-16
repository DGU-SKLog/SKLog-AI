from fastapi import FastAPI
from pydantic import BaseModel
import openai

app = FastAPI()
from pathlib import Path
import environ
import os

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

@app.post("/query")
def text(input_text: InputText):


    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text.request + input_text.content}
    ]

    response = openai.ChatCompletion.create(
        temperature=0,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']
    print("query: ", input_text.request + input_text.content)
    print("answer: ", answer)
    return answer

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)