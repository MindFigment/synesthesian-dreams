from tkinter import *
import numpy as np
from functools import partial
import tensorflow as tf
from PIL import ImageTk, Image

latent_code_size = 10

class LatentCodePanel(Frame):
    def __init__(self, parent, latent_code_size):
        Frame.__init__(self, parent)
        self.parent = parent
        self.size = latent_code_size
        self.scales = []
        self.latent = []
        self.labels = []
        self.latent_code = tf.random.normal([1, 100]).numpy()
        self.loaded = None
        self.infer = None
        self.photo = None
        self.initialize()

    def initialize(self):
        """
        Draw GUI
        """

        self.frame = Frame(self.parent)
        self.frame.grid(row=0, column=0)
        self.frame2 = Frame(self.parent)
        self.frame2.grid(row=0, column=1)
        latent_code_label = Label(self.frame, text="Latent Code")
        latent_code_label.grid(row=0, column=0)

        # Create a latent code panel
        for i in range(0, 50):
            var = DoubleVar(value=self.latent_code[0][i])
            self.latent.append(var)
            label = Label(self.frame, textvariable=var)
            self.labels.append(label)
            scale_manipulation_partial = partial(self.scale_manipulation, index=i)
            s = Scale(self.frame, variable=self.latent[i], orient=HORIZONTAL, from_=-1.0, to=1.0, resolution=0.01, showvalue=False, command=scale_manipulation_partial)
            self.scales.append(s)
            self.scales[i].grid(row=i+1,  column=0)
            self.labels[i].grid(row=i+1, column=1)


        self.loaded = tf.saved_model.load("./models/1/")
        self.infer = self.loaded.signatures["serving_default"]


    def generate_image(self):
        x = self.infer(tf.convert_to_tensor(self.latent_code, dtype=tf.float32))
        img = x["conv2d_transpose_2"].numpy().squeeze()

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


if __name__ == "__main__":

    root = Tk()
    root.title("Manipulate GAN's generated image")
    root.geometry("600x600")
    root.grid_rowconfigure(1, weight=1)
    root.grid_columnconfigure(1, weight=1)

    latent_code_panel = LatentCodePanel(root, latent_code_size)

    root.mainloop()