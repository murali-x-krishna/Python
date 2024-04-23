from datetime import datetime, timedelta
from openai import OpenAI

"""
Main program is below
"""

# Print the start time
print(Rf"The start time is {datetime.today().strftime('%Y-%m-%d %H:%M:%S:%f')}")

# Create the OpenAI Client object
client = OpenAI()

# Use a test piece of code below
completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "system", "content": "You are a poetic assistant, skilled in explaining complex programming concepts with creative flair."},
    {"role": "user", "content": "Compose a poem that explains the concept of recursion in programming."}
  ]
)

print(completion.choices[0].message)

# Print the end time
print(Rf"The end time is {datetime.today().strftime('%Y-%m-%d %H:%M:%S:%f')}")


