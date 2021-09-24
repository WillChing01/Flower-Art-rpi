import os,sys
from flask import Flask,send_file

sys.path.insert(0,os.getcwd())
from generator import *

app=Flask(__name__)

@app.route('/')
def hello():
    seed=torch.randn(16,latent_size,1,1,device=device)
    img=generator(seed)
    save_image(denorm(img),os.getcwd()+'\\0.png',nrow=4)
    return send_file('0.png',mimetype='image/png')
