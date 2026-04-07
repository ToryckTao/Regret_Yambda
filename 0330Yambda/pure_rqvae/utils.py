import os
import datetime


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def get_local_time():
    return datetime.datetime.now().strftime("%b-%d-%Y_%H-%M-%S")


def set_color(text, color, highlight=True):
    colors = ["black", "red", "green", "yellow", "blue", "pink", "cyan", "white"]
    try:
        idx = colors.index(color)
    except ValueError:
        idx = 6
    code = f"\033[{'1' if highlight else '0'};3{idx}m"
    return f"{code}{text}\033[0m"
