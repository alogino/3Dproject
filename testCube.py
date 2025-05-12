import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from numpy import asarray
import time


width = 1280  
height = 720

# Création du pipeline
from pipeline import GraphicPipeline
pipeline = GraphicPipeline(width, height)

# Configuration de la caméra
from camera import Camera
from projection import Projection
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

vertices = np.array([
    # Position (x,y,z), Normale (nx,ny,nz), UV (u,v)
    # Face +Z (avant)
    [-0.5, -0.5,  0.5,  0, 0, 1,  0, 1],  # Bas-gauche
    [ 0.5, -0.5,  0.5,  0, 0, 1,  1, 1],  # Bas-droite
    [ 0.5,  0.5,  0.5,  0, 0, 1,  1, 0],  # Haut-droite
    [-0.5,  0.5,  0.5,  0, 0, 1,  0, 0],  # Haut-gauche

    # Face -Z (arrière)
    [ 0.5, -0.5, -0.5,  0, 0,-1,  0, 1],  # Bas-droite (mirroir)
    [-0.5, -0.5, -0.5,  0, 0,-1,  1, 1],  # Bas-gauche (mirroir)
    [-0.5,  0.5, -0.5,  0, 0,-1,  1, 0],  # Haut-gauche (mirroir)
    [ 0.5,  0.5, -0.5,  0, 0,-1,  0, 0],  # Haut-droite (mirroir)

     # Face +X (droite)
    [ 0.5,  0.5, -0.5,  1, 0, 0,  0, 1],
    [ 0.5,  0.5,  0.5,  1, 0, 0,  1, 1],
    [ 0.5, -0.5,  0.5,  1, 0, 0,  1, 0],
    [ 0.5, -0.5, -0.5,  1, 0, 0,  0, 0],
    
    # Face -X (gauche)
    [-0.5, -0.5, -0.5, -1, 0, 0,  1, 0],
    [-0.5, -0.5,  0.5, -1, 0, 0,  0, 0],
    [-0.5,  0.5,  0.5, -1, 0, 0,  0, 1],
    [-0.5,  0.5, -0.5, -1, 0, 0,  1, 1],

    # Face +Y (haut) #modified
    [-0.5,  0.5,  0.5,  0, 1, 0,  0, 1],
    [ 0.5,  0.5,  0.5,  0, 1, 0,  1, 1],
    [ 0.5,  0.5, -0.5,  0, 1, 0,  1, 0],
    [-0.5,  0.5, -0.5,  0, 1, 0,  0, 0],

    # Face -Y (bas)
    [-0.5, -0.5, -0.5,  0,-1, 0,  0, 1],
    [ 0.5, -0.5, -0.5,  0,-1, 0,  1, 1],
    [ 0.5, -0.5,  0.5,  0,-1, 0,  1, 0],
    [-0.5, -0.5,  0.5,  0,-1, 0,  0, 0]
], dtype=np.float32)


triangles = np.array([
    # Face +Z
    [0, 1, 2], [0, 2, 3],
    # Face -Z
    [4, 5, 6], [4, 6, 7],
    # Face +X
    [8, 9, 10], [8, 10, 11],
    # Face -X
    [12, 13, 14], [12, 14, 15],
    # Face +Y
    [16, 17, 18], [16, 18, 19],
    # Face -Y
    [20, 21, 22], [20, 22, 23]
])


 
# Chargement des textures

cube_map_faces = {
    'posx': asarray(Image.open('posx.jpg'))[:, :, :3],
    'negx': asarray(Image.open('negx.jpg'))[:, :, :3],
    'posy': asarray(Image.open('posy.jpg'))[:, :, :3],
    'negy': asarray(Image.open('negy.jpg'))[:, :, :3],
    'posz': asarray(Image.open('posz.jpg'))[:, :, :3],
    'negz': asarray(Image.open('negz.jpg'))[:, :, :3]
}


# Données pour le pipeline
data = dict([
    ('viewMatrix', cam.getMatrix()),
    ('projMatrix', proj.getMatrix()),
    ('cameraPosition', position),
    ('lightPosition', np.array([10, 0, 10])),
    ('environment', cube_map_faces)
])

# Rendu
start = time.time()
pipeline.draw(vertices, triangles, data)
end = time.time()
print(f"Temps de rendu: {end - start:.2f} secondes")

# Affichage
plt.imshow(pipeline.image)
plt.show()
