import os
import logging


def updateConfig(obj, ref):
    if isinstance(ref, dict):
        for member, value in ref.items():
            setattr(obj, member, value)


def create_dirs(dirs):
    """
    dirs - a list of directories to create if these directories are not found
    :param dirs:
    :return:
    """
    try:
        for dir_ in dirs:
            if not os.path.exists(dir_):
                os.makedirs(dir_)
                print(f"Created dir: {dir_}")
            print(f"Dir: {dir_} already exists")
    except Exception as err:
        # logging.getLogger("Dirs Creator").info("Creating directories error: {0}".format(err))
        print(f"Creating directories error: {err}")
        exit(-1)


def yes_or_no(question):
    valid_answers = ["yes", "no", "y", "n"]
    answer = input(f"{question} (y/n): ").lower().strip()
    print("")
    while answer not in valid_answers:
        print("Input yes(y) or no(n)")
        answer = input(f"{question} (y/n): ").lower().strip()
        print("")
    if answer[0] == "y":
        return True
    else:
        return False
