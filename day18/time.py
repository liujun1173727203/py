import datetime
print(datetime.datetime.now())
print(datetime.datetime.now()-datetime.timedelta(weeks=3))
print(datetime.datetime.now().replace(year=1999,month=11,day=25))
print(datetime.date.fromtimestamp(211421424))