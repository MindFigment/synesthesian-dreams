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
