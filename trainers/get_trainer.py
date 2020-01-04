from importlib import import_module

def get_trainer(trainer):

    trainers = {"DCGAN": ("dcgan_trainer", "DCGANTrainer"), "MSGGAN": ("msggan_trainer", "MSGGANTrainer")}

    if trainer not in trainers:
        raise AttributeError(f"Invalid module name: {trainer}")

    module_name = f"trainers.{trainers[trainer][0]}"
    class_name = trainers[trainer][1]

    module = import_module(module_name)

    return getattr(module, class_name)