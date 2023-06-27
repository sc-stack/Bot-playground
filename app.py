from IPython.display import display, Markdown, Latex, HTML, JSON
import openai
import os

from dotenv import load_dotenv, find_dotenv
_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('OPENAI_API_KEY')


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=1,  # this is the degree of randomness of the model's output
    )
    return response.choices[0].message["content"]

# SUMMARIZING
# LLM's used to summarize text
# Ex: Ecommerce website - tool to summarize over reviews if most reviews are large and lengthy
# Modiy prompt to generate a summarize that is more applicable to one particular group in your business
# Try extract instead of summarize
# Loop through reviews in an array and call get_completion for each review
# Can Build stuff like a dashboard - quicksight data pipelines etl etc


text1 = f"""
The National Basketball Association (NBA) is one of the most popular professional sports leagues in the world, especially known for showcasing some of the best basketball talent. As of my knowledge cutoff in 2021, the league consists of 30 teams, divided into two conferences: the Eastern and Western. The league has seen various dynamic and skillful players over the years, such as Michael Jordan, Kobe Bryant, LeBron James, and Stephen Curry. The NBA's influence extends beyond sports, affecting fashion, music, and wider popular culture.
"""

text2 = f"""
The Ultimate Fighting Championship (UFC) is the world's leading mixed martial arts (MMA) organization. This sport combines aspects of boxing, wrestling, jiu-jitsu, kickboxing, and other disciplines. The UFC has played a key role in popularizing MMA globally, with fighters from all around the world competing in various weight classes. Superstars such as Conor McGregor, Khabib Nurmagomedov, and Amanda Nunes have helped elevate the sport's popularity to new heights.
"""

text3 = f"""
Football, also known as soccer in some countries, is the world's most popular sport. With a rich history that traces back over a century, the sport is played in nearly every corner of the globe. At the professional level, leagues like the English Premier League, Spain's La Liga, and Germany's Bundesliga are followed by millions of fans. Every four years, the FIFA World Cup brings nations together in a celebration of football, marking one of the most significant events in the sporting calendar.
"""

text4 = f"""
Hockey, particularly ice hockey, is a major sport in countries like Canada, the United States, Russia, and parts of Europe. The National Hockey League (NHL) is the premier professional ice hockey league in North America, boasting 32 teams as of 2021. Known for its speed, physicality, and strategic gameplay, hockey attracts a passionate fanbase. Great players like Wayne Gretzky, Mario Lemieux, and Sidney Crosby have left indelible marks on the sport.
"""

text5 = f"""
Baseball, often referred to as America's pastime, has a significant place in U.S. sporting culture. The Major League Baseball (MLB) is the oldest of the major professional sports leagues in the United States and Canada. A total of 30 teams play in the National League (NL) and American League (AL). Baseball is unique for its combination of strategic thinking and physical skill. Legends of the sport, such as Babe Ruth, Jackie Robinson, and Derek Jeter, have helped shape its rich history.
"""

paragraphs = [text1, text2, text3, text4, text5]

for i in range(len(paragraphs)):
    prompt = f"""
    Your task is to generate a short summary of this paragraph below delimted by backticks. For each paragraph,
    output a json object with the keys: summary, and similar-paragraph where the value for similar
    paragraph includes another paragraph written about a similar topic that is consistent with the tone
    and style of the current paragraph being analyzed. Do the summary and similar paragraph in 30 words or less

    Paragraph: ```{paragraphs[i]}```
    """

    response = get_completion(prompt)

    print(response, "\n")

# INFERRING - Model takes text as input and performs anaylsis - extract labels, names, sentiment(positive, negative)
# Ex: Classify the sentiment of a review - already see a usecase - classify based on list of sentimemt and integrate into dashboards for BI
# Can extract items made by reviewer and company that made item - can use this with BI with the sentiment
# Call get_completions 3 times OR write single prompt to extract all sentiment
# Given a long piece of text, a LLM can infer stuff about the text - what is the text about "Determine x topics discussed in the text. Format response
# as a list of items"
# Given an article which article from list of topics are included in the article below. Give your answer as list with 0 or 1 for each topic
# if topic is included - do something

for i in range(len(paragraphs)):
    topics = ["MMA", "Sports", "Basketball"]
    prompt = f"""
    Identify a list of 3 emotions the author is expressing. Format your answer as a list
    of lower-case words seperated by commas. Also, output this as a JSON object where "emotions",
    "anger", and "sport" are fields. Make sure to set anger to a boolean value. Also Output another JSON object with
    the keys corresponding to the values in {topics} and their respective values being whether the
    topic is included within the paragraph. Make this output an array of length n, where n is the number
    of JSON objects included in the output.

    Paragraph: ```{paragraphs[i]}```
    """

    response = get_completion(prompt)

    print(response, "\n")

# TRANSFORMING - LLM's are good at transforming its input to another format. Transforming language into another language
# Inputting HTML and outputting JSON. Can do multiple translations at once. Can also ask to translate in formal and informal forms
# Users can tell company, IT issues in a variety of different languages. Can loop through and translate to a unified languages
# Tone translation: Trnaslate the following from x tone to y tone
# Format Conversion: ChatGPT can translate between formats: The prompt should describe the input and output formats
# Spell-Check - Grammar checkers, proof-read and correct, change the tone
data_json = {"resturant employees": [
    {"name": "Shyam", "email": "shyamjaiswal@gmail.com"},
    {"name": "Bob", "email": "bob32@gmail.com"},
    {"name": "Jai", "email": "jai87@gmail.com"}
]}


prompt = f"""
Translate the following python dictionary from JSON to an HTML \
table with column headers and title. Make the table in chinese: {data_json}
"""
response = get_completion(prompt)
print(response)
display(HTML(response))

# EXPANDING - generate longer piece of text from shorter piece of text in LLM
# Can be used to generate a large amount of spam, use temperature to vary degree of
# expiration and variety in the model
# can use a customer service AI assistant to sent an email to reply to an customer email
# based off the sentiment of the customer email
# At a higher temperature the model will make riskier assumptions. Responses will diverge at higher
# temperatures. Use 0 temperature if you want a predictable response for applications. For tasks that
# require variety use a higher temperature
# infer sentiment
# and the original customer message, customize the email

# review for a blender
review = f"""
So, they still had the 17 piece system on seasonal \
sale for around $49 in the month of November, about \
half off, but for some reason (call it price gouging) \
around the second week of December the prices all went \
up to about anywhere from between $70-$89 for the same \
system. And the 11 piece system went up around $10 or \
so in price also from the earlier sale price of $29. \
So it looks okay, but if you look at the base, the part \
where the blade locks into place doesn’t look as good \
as in previous editions from a few years ago, but I \
plan to be very gentle with it (example, I crush \
very hard items like beans, ice, rice, etc. in the \ 
blender first then pulverize them in the serving size \
I want in the blender then switch to the whipping \
blade for a finer flour, and use the cross cutting blade \
first when making smoothies, then use the flat blade \
if I need them finer/less pulpy). Special tip when making \
smoothies, finely cut and freeze the fruits and \
vegetables (if using spinach-lightly stew soften the \ 
spinach then freeze until ready for use-and if making \
sorbet, use a small to medium sized food processor) \ 
that you plan to use that way you can avoid adding so \
much ice if at all-when making your smoothie. \
After about a year, the motor was making a funny noise. \
I called customer service but the warranty expired \
already, so I had to buy another one. FYI: The overall \
quality has gone done in these types of products, so \
they are kind of counting on brand recognition and \
consumer loyalty to maintain sales. Got it in about \
two days.
"""

prompt = f"""
You are a customer service AI assistant.
Your task is to send an email reply to a valued customer
based of the sentiment as good bad or neutral. 
Given the customer email delimited by ```, \
Generate a reply to thank the customer for their review.
If the sentiment is positive or neutral, thank them for \
their review.
If the sentiment is negative, apologize and suggest that \
they can reach out to customer service. 
Make sure to use specific details from the review.
Write in a concise and professional tone.
Sign the email as `AI customer agent`.
Output this as a JSON object with the keys "email" and "sentiment"
Customer review: ```{review}```
"""
response = get_completion(prompt)
print(response)

# CHATBOTS
# LLM's can be used to build a custom chatbot to play the role of an AI customer service agent
# or order taker for a restaurant. New get_completions function - user message is input and assistant message is output
# Messages can be from a list of roles. System message sets the behavior and persona for the conversation - provides the dev
# with a way of framing the conversation


def get_completion_from_messages(messages, model="gpt-3.5-turbo", temperature=0):
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=temperature,  # this is the degree of randomness of the model's output
    )
#     print(str(response.choices[0].message))
    return response.choices[0].message["content"]


# Need to give context in messages btw if we want bot the remember previous info
messages = [
    {'role': 'system', 'content': 'You are an assistant that speaks like Shakespeare.'},
    {'role': 'user', 'content': 'tell me a joke'},
    {'role': 'assistant', 'content': 'Why did the chicken cross the road'},
    {'role': 'user', 'content': 'I don\'t know'}]

# response is assistant message
response = get_completion_from_messages(messages, temperature=1)
print(response)


# ITERATIVE PROOMPT DEVELOPMENT
# Idea -> Implementation (Code/Data) -> Experimental result -> Error Analysis
# Be Clear and specific, analyze why result does not give desired output, refine the idea and the prompt, repeat
# Error Ex: Text too long, Text response focuses on the wrong details
