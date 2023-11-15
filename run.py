import openai
import pandas as pd
import matplotlib.pyplot as plt

def question(query):

    # FTA API
    API_KEY = "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"
    openai.api_key = API_KEY

    model = "gpt-3.5-turbo"  

    messages = [
        {"role":"system", "content":"You are a helpful assistant."},
        {"role":"user","content":query}
    ]

    response = openai.ChatCompletion.create(
        temperature=0,
        model=model,
        messages = messages
    )

    answer = response['choices'][0]['message']['content']

    return answer


question_query = f"""
1. Question: Please make the text below into a markdown bullet point

"""

rule_query = """
You only need to answer the bullet point in markdown format, and if the bullet point is not in a format or cannot be changed to a markdown bullet point, print out the reason as an example below.

Example1:
프로세스와 스레드
프로그램은 명령어가 실행되는 순서의 집합을 의미한다.
프로세스는 운영체제에 의해 현재 실행 중인 프로그램이며, 운영체제로부터 자원을 할당받아 실행된다.
스레드는 프로세스 내에서 실행되는 작업의 단위를 뜻한다.
->
## 프로세스와 스레드

- 프로그램: 명령어가 실행되는 순서의 집합
- 프로세스: 운영체제에 의해 현재 실행 중인 프로그램이며, 운영체제로부터 자원을 할당받아 실행
- 스레드: 프로세스 내에서 실행되는 작업의 단위
"""
result_query = """
2.Answer : 
You should only print out the results, and never say plain text except when conversion to markdown is not possible.
"""

text_data = """덧셈과 뺄셈
덧셈은 두 개 이상의 숫자를 합하는 연산이며, 결과는 합계입니다.
뺼셈은 한 수에서 다른 수를 빼는 연산이며, 결과는 차입니다.""" # 표로 만들 평문

context_query = "The plain text to be made in the form of a markdown table is as follows. \n\n"  + text_data

query1 = question_query + rule_query + result_query + context_query

print(query1)


ans1 = question(query1)
print(ans1)
