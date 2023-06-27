import os
import openai
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]

# Evaluate outputs by system - use moderation API, and other things to evaluate output quality before


def get_completion_from_messages(
    messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return response.choices[0].message["content"]

# 1. see if input is flagged by moderation API
# 2. Extract list of products
# 3. Look up products
# 4. Answer User question
# 5. Put through answer 

