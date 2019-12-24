import click
import yaml
from easydict import EasyDict as edict
from pprint import pprint

import logging
from logging import Formatter
from logging.handlers import RotatingFileHandler

from utils.utils import create_dirs

import os



def CommandWithConfigFile(config_name):

    class CustomCommandClass(click.Command):

        def invoke(self, ctx):
            config_file = ctx.params[config_name]
            config = process_config(config_file)
                
            ctx.params[config_name] = edict(config)

            return super(CustomCommandClass, self).invoke(ctx)
                        
    return CustomCommandClass


def setup_logging(log_dir):
    log_file_format = "[%(levelname)s] - %(asctime)s - %(name)s - : %(message)s in %(pathname)s:%(lineno)d"
    log_console_format = "[%(levelname)s]: %(message)s"

    # Main logger
    main_logger = logging.getLogger()
    main_logger.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(Formatter(log_console_format))

    exp_file_handler = RotatingFileHandler('{}exp_debug.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_file_handler.setLevel(logging.DEBUG)
    exp_file_handler.setFormatter(Formatter(log_file_format))

    exp_errors_file_handler = RotatingFileHandler('{}exp_error.log'.format(log_dir), maxBytes=10**6, backupCount=5)
    exp_errors_file_handler.setLevel(logging.WARNING)
    exp_errors_file_handler.setFormatter(Formatter(log_file_format))

    main_logger.addHandler(console_handler)
    main_logger.addHandler(exp_file_handler)
    main_logger.addHandler(exp_errors_file_handler)


def get_config_from_yaml(yaml_file):
    """
    .
    """

    with open(yaml_file, "r") as config_file:
        try:
            config_dict = yaml.safe_load(config_file)
            config = edict(config_dict)
            return config
        except ValueError:
            print("Invalid YAML file format... Please provide good yaml file")
            exit(-1)


def process_config(yaml_file):
    """
    .
    """

    config = get_config_from_yaml(yaml_file)
    print("THE CONFIGURATION of this experiment")
    pprint(config)

    try:
        print(70 * "-")
        print(f"The name of this experiment is: {config.exp_name}")
        print(70 * "-")
    except AttributeError:
        print("ERROR!! Please provide the exp_name in yaml config file...")
        exit(-1)

    # Setup some important directories to be used for that run
    dirs_to_create = []

    if config.t_c.want_log:
        config.t_c.summary_dir = os.path.join("experiments", config.exp_name, "summaries")
        dirs_to_create += [ config.t_c.summary_dir ]

    config.t_c.checkpoint_dir = os.path.join("experiments", config.exp_name, "checkpoints")
    config.t_c.out_dir = os.path.join("experiments", config.exp_name, "out")
    config.t_c.log_dir = os.path.join("experiments", config.exp_name, "logs")

    dirs_to_create += [ config.t_c.checkpoint_dir, config.t_c.out_dir, config.t_c.log_dir ]

    create_dirs(dirs_to_create)

    return config

    # Setup logging in the project
    # setup_logging(config.log_dir)

    # logging.getLogger().info("Hi, This is root.")
    # logging.getLogger().info("After the configurations are successfully processed and dirs are created.")
    # logging.getLogger().info("The pipeline of the project will begin now.")