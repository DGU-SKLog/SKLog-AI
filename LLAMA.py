from fastapi import FastAPI
from pydantic import BaseModel
from typing import List, Optional
import openai
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, GenerationConfig
from peft import LoraConfig, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training
import warnings
import torch
#모델 불러오기
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

app = FastAPI()
from pathlib import Path
# import environ
import os
import json

device = (
        torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    )
model_id = "kfkas/Llama-2-ko-7b-Chat"
model = AutoModelForCausalLM.from_pretrained(
    model_id, device_map={"": 0},torch_dtype=torch.float16, low_cpu_mem_usage=True
)
tokenizer = AutoTokenizer.from_pretrained(model_id)
model.eval()
model.config.use_cache = (True)
tokenizer.pad_token = tokenizer.eos_token


def gen(x, model, tokenizer, device):
    prompt = (
        f"아래는 작업을 설명하는 명령어입니다. 요청을 적절히 완료하는 응답을 작성하세요.\n\n### 명령어:\n{x}\n\n### 응답:"
    )
    len_prompt = len(prompt)
    gened = model.generate(
        **tokenizer(prompt, return_tensors="pt", return_token_type_ids=False).to(
            device
        ),
        max_new_tokens=1024,
        early_stopping=True,
        do_sample=True,
        top_k=20,
        top_p=0.9,
        no_repeat_ngram_size=3,
        eos_token_id=2,
        repetition_penalty=1.2,
        num_beams=3
    )
    len_prompt=tokenizer.decode(gened[0]).index('응답:')+3
    return tokenizer.decode(gened[0])[len_prompt:-4]

def LLM_infer(input):

    output = gen(input, model=model, tokenizer=tokenizer, device=device)

    return output


# if __name__ == "__main__":
#     text = LLM_infer("삼원색에 대해 알려줘")
#     print(text)

# #실행방법 uvicorn main:app --reload
# # Build paths inside the project like this: BASE_DIR / 'subdir'.
# model_id = "beomi/llama-2-ko-7b"
# bnb_config = BitsAndBytesConfig(  # 4비트로 양자화 // 훈련만 하고 이번 주 발표x 다음 주에 소개
#     load_in_4bit=False,
#     bnb_4bit_use_double_quant=False,
#     # bnb_4bit_quant_type="nf4",
#     # bnb_4bit_compute_dtype=torch.bfloat16
# )
#
# tokenizer = AutoTokenizer.from_pretrained(model_id,device_map="auto")  # 토크나이저 자동으로 잡아줌, llama로 지정해서 해도 무관함.
# # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=bnb_config, device_map={"": 0})
# model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
# #모델 불러오
#
# from transformers import AutoModelForCausalLM, AutoTokenizer






class InputText(BaseModel):
    request: str
    content: str
    userContentExamples: Optional[List[str]] = None
    aiContentExamples: Optional[List[str]] = None

class ChatText(BaseModel):
    question: str

class CohensionText(BaseModel):
    content: str

class MetaText(BaseModel):
    content: str


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



    # messages = [
    #     {"role": "system", "content": "You are a robot that only outputs the results of requests without any interaction with the questioner."},
    #     {"role": "user", "content": query}
    # ]

    # response = openai.ChatCompletion.create(
    #     temperature=0.8,
    #     model=model,
    #     messages=messages
    # )

    # answer = response['choices'][0]['message']['content']
    print("*query: ", query)
    answer = LLM_infer(query)
    print("*answer: \n", answer)
    return {'content': answer}




@app.post("/ai/cohesion")
def cohension(co_text: CohensionText):

    query = f"""
You should only print the results for the question without plaintext.
You just have to answer the "query". DO NOT attempt any other interaction.
Do your best to carry out the command. 
If the output cannot be generated or is ambiguous, you should just print "해당 작업에 대한 정보가 부족합니다. 더 정확한 데이터를 입력해주세요".

question is = Please edit the provided content for uniformity.

You only need to answer the uniformed text, and if the uniformed text are not provided or does not make sense, print out the reason as an example below.
Please respond in the same language as the content; if the content is in Korean, respond in Korean, and if it's in English, respond in English.
provided content is = {co_text.content}
"""


    # messages = [
    #     {"role": "system", "content": "You are a robot that only outputs the results of requests without any interaction with the questioner."},
    #     {"role": "user", "content": query}
    # ]
    #
    # response = openai.ChatCompletion.create(
    #     temperature=0,
    #     model=model,
    #     messages=messages
    # )
    answer = LLM_infer(query)
    # answer = response['choices'][0]['message']['content']

    print("*query: ", query)
    print("*answer: \n", answer)
    return {'content': answer}




@app.post("/ai/metadata")
def metaData(me_text: MetaText):

    query = f"""
You should only print the results for the question without plaintext.
You just have to answer the "query". DO NOT attempt any other interaction.
Do your best to carry out the command. 
If the output cannot be generated or is ambiguous, you should just print "해당 작업에 대한 정보가 부족합니다. 더 정확한 데이터를 입력해주세요".

question is = Please generate one title and three tags that form of dictionary based on the following content.

You only need to answer the title and tags form of dictionary, and if the title and tags are not provided or does not make sense, print out the reason as an example below.
Please respond in the same language as the content; if the content is in Korean, respond in Korean, and if it's in English, respond in English.
Dictionary form should be {{
    title: 'Title that you make', 
tags: ['tag1', 'tag2', 'tag3']
}}

following content is = {me_text.content}
"""

    # messages = [
    #     {"role": "system", "content": "You are a robot that only outputs the results of requests without any interaction with the questioner."},
    #     {"role": "user", "content": query}
    # ]
    #
    # response = openai.ChatCompletion.create(
    #     temperature=0,
    #     model=model,
    #     messages=messages
    # )
    #
    # answer = response['choices'][0]['message']['content']
    answer = LLM_infer(query)
    answer_dict = json.loads(answer) ##answer가 string으로 되어있으므로 json형식으로 변환
    title = answer_dict.get('title', None) ##dictionary로 변환
    tags = answer_dict.get('tags', None) ##dictionary로 변환

    print("*query: ", query)
    print("*answer: \n", answer)
    return {'title': title, 'tags': tags}


@app.post("/ai/chat-bot")
def chat_bot(chat_text: ChatText):
    query = chat_text.question

    print("query: ", query)

    messages = [
        {"role": "system", "content": "you are helpful assistant."},
        {"role": "user", "content": query}
    ]

    # response = openai.ChatCompletion.create(
    #     temperature=0,
    #     model=model,
    #     messages=messages
    # )
    #
    # answer = response['choices'][0]['message']['content']
    answer = LLM_infer(query)
    print("answer: ", answer)
    return {'answer': answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)