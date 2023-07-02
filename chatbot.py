# Building Complex Systems using LLMs (Chat Gpt API)
# Supervised Learning Used to Train LLM in order to repeatedly predict the next word
# TWO LLM's - Base LLM, Instruction Tuned LLM
# Train a Base LLM -> Instruction Tuned LLM
# LLM's repeteadely predicts the next token not next word - hence why asking chatgpt to reverse a word
# does not give u desired output unless you add dashes between the letters
# Input = context, Output = Completion
# First classify the inputs - then give secondary instructions based on the primary instructions

import os
import openai

# import tiktoken
from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())  # read local .env file

openai.api_key = os.environ["OPENAI_API_KEY"]


def get_completion_from_messages(
    messages, model="gpt-3.5-turbo", temperature=0, max_tokens=500
):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
        max_tokens=max_tokens,  # the maximum number of tokens the model can ouptut
    )
    return response.choices[0].message["content"]


# System message sets the tone of what you want the LLM to do
# User message is what you want the LLM to do
# Assistant message is the LLM response
# Can also input assistant message to continue convo based on things previously said
# messages = [
#     {'role': 'system',
#      'content': """You are an assistant who\
#  responds in the style of Dr Seuss."""},
#     {'role': 'user',
#      'content': """write me a very short poem\
#  about a happy carrot"""},
# ]
# response = get_completion_from_messages(messages, temperature=1)
# print(response)

# # combined
# messages = [
#     {'role': 'system',
#      'content': """You are an assistant who \
# responds in the style of Dr Seuss. \
# All your responses must be one sentence long."""},
#     {'role': 'user',
#      'content': """write me a story about a happy carrot"""},
# ]
# response = get_completion_from_messages(messages,
#                                         temperature=1)
# print(response)


# def get_completion_and_token_count(messages,
#                                    model="gpt-3.5-turbo",
#                                    temperature=0,
#                                    max_tokens=500):

#     response = openai.ChatCompletion.create(
#         model=model,
#         messages=messages,
#         temperature=temperature,
#         max_tokens=max_tokens,
#     )

#     content = response.choices[0].message["content"]

#     token_dict = {
#         'prompt_tokens': response['usage']['prompt_tokens'],
#         'completion_tokens': response['usage']['completion_tokens'],
#         'total_tokens': response['usage']['total_tokens'],
#     }

#     return content, token_dict


# messages = [
#     {'role': 'system',
#      'content': """You are an assistant who responds\
#  in the style of Dr Seuss."""},
#     {'role': 'user',
#      'content': """write me a very short poem \
#  about a happy carrot"""},
# ]
# response, token_dict = get_completion_and_token_count(messages)

# print(response)

# print(token_dict)

# delimiter = "####"
# system_message = f"""
#     You will be provided with customer service queries. \
#     The customer service query will be delimited with \
#     {delimiter} characters.
#     Classify each query into a primary category \
#     and a secondary category.
#     Provide your output in json format with the \
#     keys: primary and secondary.

#     Primary categories: Billing, Technical Support, \
#     Account Management, or General Inquiry.

#     Billing secondary categories:
#     Unsubscribe or upgrade
#     Add a payment method
#     Explanation for charge
#     Dispute a charge

#     Technical Support secondary categories:
#     General troubleshooting
#     Device compatibility
#     Software updates

#     Account Management secondary categories:
#     Password reset
#     Update personal information
#     Close account
#     Account security

#     General Inquiry secondary categories:
#     Product information
#     Pricing
#     Feedback
#     Speak to a human

# """
# # Can then process this data. Use output from this step as input to a subsequent step
# # Based on categorization of a customer inquiry we can do other steps
# user_message = f"""\
# I want you to delete my profile and all of my user data"""
# messages = [
#     {"role": "system", "content": system_message},
#     {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"},
# ]
# response = get_completion_from_messages(messages)
# print(response)

# user_message = f"""\
# Tell me more about your flat screen tvs"""
# messages = [
#     {"role": "system", "content": system_message},
#     {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"},
# ]
# response = get_completion_from_messages(messages)
# print(response)

# # Evaluate Inputs - Moderation
# # Use openAI moderation API

# response = openai.Moderation.create(
#     input="""
# Here's the plan.  We get the warhead,
# and we hold the world ransom...
# ...FOR ONE MILLION DOLLARS!
# """
# )
# moderation_output = response["results"][0]
# print(moderation_output)

# Avoid prompt injections - when user attempts to manipulate AI systems
# Importsant to detect and prevent them - use delimters and clear instructions
# or use an additional prompt to see if the user is trying to carry out prompt injection
# Remove any delimter characters in the user message in case user asks what are your delimeter character

# Tasks that generate an input and generate an output based on previous steps: Reframe of query to
# requests a series of reasoning steps: Called chain of thought reasoning. Hide model's reasoning
# from the user - inner monologue

delimiter = "####"
system_message = f"""
Follow these steps to answer the customer queries.
The customer query will be delimited with four hashtags,\
i.e. {delimiter}. 

Step 1:{delimiter} First decide whether the user is \
asking a question about a specific product or products. \
Product cateogry doesn't count. 

Step 2:{delimiter} If the user is asking about \
specific products, identify whether \
the products are in the following list.
All available products: 
1. Product: TechPro Ultrabook
   Category: Computers and Laptops
   Brand: TechPro
   Model Number: TP-UB100
   Warranty: 1 year
   Rating: 4.5
   Features: 13.3-inch display, 8GB RAM, 256GB SSD, Intel Core i5 processor
   Description: A sleek and lightweight ultrabook for everyday use.
   Price: $799.99

2. Product: BlueWave Gaming Laptop
   Category: Computers and Laptops
   Brand: BlueWave
   Model Number: BW-GL200
   Warranty: 2 years
   Rating: 4.7
   Features: 15.6-inch display, 16GB RAM, 512GB SSD, NVIDIA GeForce RTX 3060
   Description: A high-performance gaming laptop for an immersive experience.
   Price: $1199.99

3. Product: PowerLite Convertible
   Category: Computers and Laptops
   Brand: PowerLite
   Model Number: PL-CV300
   Warranty: 1 year
   Rating: 4.3
   Features: 14-inch touchscreen, 8GB RAM, 256GB SSD, 360-degree hinge
   Description: A versatile convertible laptop with a responsive touchscreen.
   Price: $699.99

4. Product: TechPro Desktop
   Category: Computers and Laptops
   Brand: TechPro
   Model Number: TP-DT500
   Warranty: 1 year
   Rating: 4.4
   Features: Intel Core i7 processor, 16GB RAM, 1TB HDD, NVIDIA GeForce GTX 1660
   Description: A powerful desktop computer for work and play.
   Price: $999.99

5. Product: BlueWave Chromebook
   Category: Computers and Laptops
   Brand: BlueWave
   Model Number: BW-CB100
   Warranty: 1 year
   Rating: 4.1
   Features: 11.6-inch display, 4GB RAM, 32GB eMMC, Chrome OS
   Description: A compact and affordable Chromebook for everyday tasks.
   Price: $249.99

Step 3:{delimiter} If the message contains products \
in the list above, list any assumptions that the \
user is making in their \
message e.g. that Laptop X is bigger than \
Laptop Y, or that Laptop Z has a 2 year warranty.

Step 4:{delimiter}: If the user made any assumptions, \
figure out whether the assumption is true based on your \
product information. 

Step 5:{delimiter}: First, politely correct the \
customer's incorrect assumptions if applicable. \
Only mention or reference products in the list of \
5 available products, as these are the only 5 \
products that the store sells. \
Answer the customer in a friendly tone.

Use the following format:
Step 1:{delimiter} <step 1 reasoning>
Step 2:{delimiter} <step 2 reasoning>
Step 3:{delimiter} <step 3 reasoning>
Step 4:{delimiter} <step 4 reasoning>
Response to user:{delimiter} <response to customer>

Make sure to include {delimiter} to separate every step.
"""

user_message = f"""
by how much is the BlueWave Chromebook more expensive \
than the TechPro Desktop"""

# response = openai.Moderation.create(
#     input="""
# Here's the plan.  We get the warhead,
# and we hold the world ransom...
# ...FOR ONE MILLION DOLLARS!
# """
# )

messages = [
    {"role": "system", "content": system_message},
    {"role": "user", "content": f"{delimiter}{user_message}{delimiter}"},
]

response = get_completion_from_messages(messages)

# Hide the chain of thought reasoning from the final output that the user sees

try:
    final_response = response.split(delimiter)[-1].strip()
except Exception as e:
    final_response = (
        "Sorry, I'm having trouble right now, please try asking another question."
    )

print(final_response)
