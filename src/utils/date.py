import datetime


def get_hour_min():
    return datetime.datetime.now().strftime('%I:%M')
