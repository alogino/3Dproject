import numpy as np
def readply(filename):
    file = open(filename, 'r')
    lines = file.readlines()
    vertices = []
    faces = []  # Changed from triangles to handle any polygon
    
    nbVertices = 0
    nbFaces = 0
    state = 0
    counter = 0
    
    for line in lines:
        if state == 0:  # Header parsing
            line = line.rstrip().split(' ')
            if line[0] == 'element':
                if line[1] == 'vertex':
                    nbVertices = int(line[2])
                if line[1] == 'face':
                    nbFaces = int(line[2])
            if line[0] == 'end_header':
                state = 1
                continue
        
        elif state == 1:  # Reading vertices
            line = line.split(' ')
            vertex = [float(l) for l in line]
            vertices.append(vertex)
            counter += 1
            if counter == nbVertices:
                state = 2
                continue
        
        elif state == 2:  # Reading faces
            parts = line.split()
            vertex_count = int(parts[0])
            vertex_indices = [int(i) for i in parts[1:1+vertex_count]]
            
            # Convert polygons to triangles (fan triangulation)
            for i in range(1, vertex_count-1):
                faces.append([vertex_indices[0], vertex_indices[i], vertex_indices[i+1]])
            counter += 1
    
    return np.array(vertices), np.array(faces, dtype=int)