from datetime import datetime, timedelta
from openai import OpenAI

"""
Function below gets me the local time. It also test string parsing etc.
"""
def getLocalTime() -> None:
    testTime = datetime.strptime("2024-04-03 07:11:05:123456", "%Y-%m-%d %H:%M:%S:%f")
    testTimeStr = testTime.strftime('%Y-%m-%d %H:%M:%S:%f')
    print(Rf"The testTime is {testTimeStr}")

    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S:%f')
    print(Rf"The now time is {now}")
    
"""
Main program is below
"""
getLocalTime()  # Dummy function call that tests local time

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

