from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import openai

app = FastAPI()
from pathlib import Path
import environ
import os

# Build paths inside the project like this: BASE_DIR / 'subdir'.
BASE_DIR = Path(__file__).resolve().parent.parent

env = environ.Env(
    API_KEY=(str, ''),
)
environ.Env.read_env(
    env_file=os.path.join(BASE_DIR,'.env')
)

API_KEY = env.API_KEY
openai.api_key = API_KEY

model = "gpt-3.5-turbo"

class InputText(BaseModel):
    text: str

@app.post("/query")
def text(input_text: InputText):

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": input_text.text}
    ]

    response = openai.ChatCompletion.create(
        temperature=0,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']

    return {"output": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)