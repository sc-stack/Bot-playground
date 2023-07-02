import os
import openai
import json

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]


# Sample "API" call the get the location of the world series
def get_location_world_series(year):
    location = {"location": "location", "year": year}
    return json.dumps(location)


functions = [
    {
        "name": "get_location_world_series",
        "description": "Gets the location of the world series in a given year",
        "parameters": {
            "type": "object",
            "properties": {
                "year": {
                    "type": "integer",
                    "description": "The year the world series was played in",
                }
            },
            "required": ["year"],
        },
    }
]

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": "Who won the world series in 2020?"},
    {"role": "assistant", "content": "The LA Dodgers won the world series in 2020"},
    {"role": "user", "content": "Where was it played?"},
]

response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo-0613",
    messages=messages,
    functions=functions,
    function_call="auto",
)

response_message = response["choices"][0]["message"]

print(response_message)

if response_message.get("function_call"):
    available_functions = {"get_location_world_series": get_location_world_series}
    function_name = response_message["function_call"]["name"]
    function_to_call = available_functions[function_name]
    function_args = json.loads(response_message["function_call"]["arguments"])
    function_response = function_to_call(year=function_args.get("year"))

    messages.append(response_message)
    messages.append(
        {"role": "function", "name": function_name, "content": function_response}
    )
    second_response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-0613", messages=messages
    )
    print(second_response["choices"][0]["message"]["content"])


# Populate DDB 
