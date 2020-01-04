import torch.nn as nn

# Custom weights initialization called netG and netG
def weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find("Block") != -1:
        for param in m.named_parameters():
            # print(type(param))
            # if type(param) == tuple:
            #     print(f"1: {param[0]}")
            #     print(f"2: {param[1]}")
            weights_init(param)
    elif classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        # nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        # print(f"init conv")
    elif classname.find('BatchNorm') != -1:
        # nn.init.xavier_uniform_(m.weight.data, gain=1.0)
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
        # print(f"init batch norm")
    # elif classname.find("tuple") != -1:
        # print("found tuple!", m[m[0]])
        # print(m[0].__class__.__name__)
        # print(m[0].weight.data)
