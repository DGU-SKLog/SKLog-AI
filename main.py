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
    userContentExamples: Optional[List[str]] = None
    aiContentExamples: Optional[List[str]] = None

@app.post("/query")
def text(input_text: InputText):

    query = f"""
You should only print the results for the question without plaintext. 
You just have to answer the "question". DO NOT attempt any other interaction.
If the output cannot be generated or is ambiguous, you should just print "Impossible"
{input_text.request} 
"""

    if input_text.userContentExamples:
        query += "The query below is an example of a query:\n"

        for user_item, ai_item in zip(input_text.userContentExamples, input_text.aiContentExamples):
            query += f"""
query:
{user_item}
        
answer:
{ai_item}
            """

    query += f"""
{input_text.content}
"""



    messages = [
        {"role": "system", "content": "You are a robot that only outputs the results of requests without any interaction with the questioner."},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        temperature=0,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']
    print("*query: ", query)
    print("*answer: \n", answer)
    return {'content': answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)