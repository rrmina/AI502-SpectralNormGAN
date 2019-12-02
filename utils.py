import numpy
import torch
import matplotlib.pyplot as plt

def ttoi(tensor):
    img = tensor.cpu().numpy()
    return img

def show(img, title=""):
    img = img.transpose(1,2,0)
    fig = plt.figure(figsize=(10,10))
    plt.title(title)
    plt.imshow(img)
    plt.show()
    plt.close()

def saveimg(image, savepath):
    image = image.transpose(1,2,0)
    plt.imsave(savepath, image)