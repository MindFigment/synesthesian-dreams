from importlib import import_module
import click
from utils.config import CommandWithConfigFile
import os
from trainers.get_trainer import get_trainer

@click.command(cls=CommandWithConfigFile("config"))
@click.option("--config", type=click.Path())
# @click.option("--m_c", type=click.Path())
# @click.option("--m_name", default="base_model")
def train(config):
    trainer = get_trainer(config.trainer)

    # Create folder if save_model is True, if folder exists it's ok
    # model_checkpoint_dir = os.path.join(config.t_c.save_folder, config.m_c.name)
    # if not os.path.exists(model_checkpoint_dir):
    #     os.makedirs(model_checkpoint_dir)
    #     print(f"!CREATED! new model directory: {model_checkpoint_dir}!")
    # else:
    #     print(f"!WARNING! directory already exists: {model_checkpoint_dir}!")

    t = trainer(config.t_c, config.m_c)
    t.run()


if __name__ == "__main__":
    train()