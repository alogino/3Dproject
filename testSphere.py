import numpy as np
from PIL import Image
from numpy import asarray
from camera import Camera
from projection import Projection
from pipeline import GraphicPipeline

width = 1280  
height = 720

import time

start = time.time()


pipeline = GraphicPipeline(width,height)


position = np.array([1.1,1.1,1.1])
lookAt = np.array([-0.577,-0.577,-0.577])
up = np.array([0.33333333,  0.33333333, -0.66666667])
right = np.array([-0.57735027,  0.57735027,  0.])


cam = Camera(position, lookAt, up, right)

nearPlane = 0.1
farPlane = 10.0
fov = 1.91986
aspectRatio = width/height

proj = Projection(nearPlane ,farPlane,fov, aspectRatio) 


lightPosition = np.array([10,0,10])

from readply import readply

vertices, triangles = readply('sphere.ply')


cube_map_faces = {
    'posx': asarray(Image.open('posx.jpg'))[:, :, :3],
    'negx': asarray(Image.open('negx.jpg'))[:, :, :3],
    'posy': asarray(Image.open('posy.jpg'))[:, :, :3],
    'negy': asarray(Image.open('negy.jpg'))[:, :, :3],
    'posz': asarray(Image.open('posz.jpg'))[:, :, :3],
    'negz': asarray(Image.open('negz.jpg'))[:, :, :3]
}


data = dict([

  ('viewMatrix',cam.getMatrix()),
  ('projMatrix',proj.getMatrix()),
  ('cameraPosition',position),
  ('lightPosition',lightPosition),
  ('environment' , cube_map_faces),

])

start = time.time()

pipeline.draw(vertices, triangles, data)


end = time.time()
print(end - start)


import matplotlib.pyplot as plt
imgplot = plt.imshow(pipeline.image)
plt.show()
