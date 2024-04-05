from datetime import datetime, timedelta

def getLocalTime() -> None:
    testTime = datetime.strptime("2024-04-03 07:11:05:123456", "%Y-%m-%d %H:%M:%S:%f")
    testTimeStr = testTime.strftime('%Y-%m-%d %H:%M:%S:%f')
    print(Rf"The testTime is {testTimeStr}")

    now = datetime.today().strftime('%Y-%m-%d %H:%M:%S:%f')
    print(Rf"The now time is {now}")
    print (Rf"This is a test... really!")

    print(Rf"I am creating a new branch")
    print("I am now testing merging and publish")
    print('I am really done now')

getLocalTime()