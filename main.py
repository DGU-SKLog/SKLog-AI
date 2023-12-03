from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai

app = FastAPI()
from pathlib import Path
import environ
import os

#실행방법 uvicorn main:app --reload
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
You just have to answer the "query". DO NOT attempt any other interaction.
Do your best to carry out the command. 
If the output cannot be generated or is ambiguous, you should just print "해당 작업에 대한 정보가 부족합니다. 더 정확한 데이터를 입력해주세요".
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
        temperature=0.8,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']
    print("*query: ", query)
    print("*answer: \n", answer)
    return {'content': answer}

class cohensionText(BaseModel):
    content: str

@app.post("/ai/cohesion")
def cohension(co_text: cohensionText):

    query = f"""
You should only print the results for the question without plaintext.
You just have to answer the "query". DO NOT attempt any other interaction.
Do your best to carry out the command. 
If the output cannot be generated or is ambiguous, you should just print "해당 작업에 대한 정보가 부족합니다. 더 정확한 데이터를 입력해주세요".

Please edit the provided text for uniformity.

You only need to answer the uniformed text, and if the uniformed text are not provided or does not make sense, print out the reason as an example below.
Please respond in the same language as the content; if the content is in Korean, respond in Korean, and if it's in English, respond in English.
{co_text.content}
"""

    messages = [
        {"role": "system", "content": "You are a robot that only outputs the results of requests without any interaction with the questioner."},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        temperature=0.8,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']
    print("*query: ", query)
    print("*answer: \n", answer)
    return {'content': answer}

class metaText(BaseModel):
    content: str
@app.post("/ai/metadata")
def cohension(me_text: metaText):

    query = f"""
You should only print the results for the question without plaintext.
You just have to answer the "query". DO NOT attempt any other interaction.
Do your best to carry out the command. 
If the output cannot be generated or is ambiguous, you should just print "해당 작업에 대한 정보가 부족합니다. 더 정확한 데이터를 입력해주세요".

Please generate titles and tags based on the following content.

You only need to answer the title and tags form of dictionary, and if the title and tags are not provided or does not make sense, print out the reason as an example below.
Please respond in the same language as the content; if the content is in Korean, respond in Korean, and if it's in English, respond in English.
Dictionary form is {title: 'Title that you make', tag: ['tag1', 'tag2', ...]}
{me_text.content}
"""

    messages = [
        {"role": "system", "content": "You are a robot that only outputs the results of requests without any interaction with the questioner."},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        temperature=0.8,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']
    print("*query: ", query)
    print("*answer: \n", answer)
    return {'content': answer}

class ChatText(BaseModel):
    question: str


@app.post("/ai/chat-bot")
def chat_bot(chat_text: ChatText):
    query = chat_text.question

    print("query: ", query)

    messages = [
        {"role": "system", "content": "you are helpful assistant."},
        {"role": "user", "content": query}
    ]

    response = openai.ChatCompletion.create(
        temperature=0.8,
        model=model,
        messages=messages
    )

    answer = response['choices'][0]['message']['content']

    print("answer: ", answer)
    return {'answer': answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)