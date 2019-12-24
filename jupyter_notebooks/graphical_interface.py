from tkinter import *
import numpy as np
from functools import partial
import torch
from PIL import ImageTk, Image
from models import Generator
import click

class LatentCodePanel(Frame):
    def __init__(self, parent, model_path):
        Frame.__init__(self, parent)
        self.parent = parent
        self.model_path = model_path
        self.scales = []
        self.latent = []
        self.labels = []
        self.device = torch.device("cuda:0" if (torch.cuda.is_available()) else "cpu")
        print(self.device)
        self.photo = None
        self.initialize()

    def initialize(self):
        """
        Draw GUI
        """

        # Load Model
        checkpoint = torch.load("".join([self.model_path, ".pt"]), map_location=self.device)
        self.latent_vector_size = 100
        netG_params = {
            "latent_vector_size": 100,
            "feature_maps_size": 64,
            "channels_num": 3
        }
        # Crate and load generator
        self.netG = Generator(**netG_params).to(self.device)
        self.netG.load_state_dict(checkpoint["netG_state_dict"])
        print(self.netG)
        self.netG.eval()
        self.latent_code = torch.randn(1, self.latent_vector_size, 1, 1, device=self.device)

        self.frame = Frame(self.parent)
        self.frame.grid(row=0, column=0)
        self.frame2 = Frame(self.parent)
        self.frame2.grid(row=0, column=1)
        latent_code_label = Label(self.frame, text="Latent Code")
        latent_code_label.grid(row=0, column=0)

        # Create a latent code panel
        for i in range(0, 10):
            var = DoubleVar(value=self.latent_code[0][i])
            self.latent.append(var)
            label = Label(self.frame, textvariable=var)
            self.labels.append(label)
            scale_manipulation_partial = partial(self.scale_manipulation, index=i)
            s = Scale(self.frame, variable=self.latent[i], orient=HORIZONTAL, from_=-1.0, to=1.0, resolution=0.01, showvalue=False, command=scale_manipulation_partial)
            self.scales.append(s)
            self.scales[i].grid(row=i+1,  column=0)
            self.labels[i].grid(row=i+1, column=1)


    def generate_image(self):
        img = self.netG(self.latent_code).detach().numpy().squeeze()

        img = np.transpose(img, (1, 2, 0))

        canvas = Canvas(self.frame2, width = 256, height = 256) 
        canvas.grid(row=0, column=1)
        img *= 255
        img += 127.5
        img = img.astype("uint8")
        img = Image.fromarray(img)
        img = img.resize((256, 256))
        self.photo = ImageTk.PhotoImage(image=img)
        canvas.create_image(20, 20, anchor=NW, image=self.photo)
        
    def scale_manipulation(self, selection, index):
        print("Index: {}, selection: {}".format(index, selection))
        self.latent_code[0][index] = float(selection)
        self.generate_image()

@click.command()
@click.option("--mn", default="first_model", help="Give the name of the model")
@click.option("--mv", default="0", help="Give the name of the model")
def main(mn, mv):
    root = Tk()
    root.title("Manipulate GAN's generated image")
    root.geometry("600x600")
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(1, weight=1)

    model_path = "".join(["./models/", mn, "_", mv])
    print(f"Model path: {model_path}")
    _ = LatentCodePanel(root, model_path)

    root.mainloop()

if __name__ == "__main__":
    main()