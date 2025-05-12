import numpy as np
#For cube and sphere

def sample(texture, u, v):
    # Clamp les valeurs entre 0 et 1
    u = max(0.0, min(1.0, u))
    v = max(0.0, min(1.0, v))
    
    # Conversion 
    u_pixel = int(u * (texture.shape[1] - 1))
    v_pixel = int((1-v) * (texture.shape[0] - 1))
    
    return texture[v_pixel, u_pixel] /255.0 


def convert_to_cubemap_coords(env_dir):
    x, y, z = env_dir
    abs_x, abs_y, abs_z = abs(x), abs(y), abs(z)
    
    if abs_x >= abs_y and abs_x >= abs_z:
        face = 'posx' if x > 0 else 'negx'
        max_axis = abs_x
        u = -y / x
        v = (z if x > 0 else -z) / x
    elif abs_y >= abs_x and abs_y >= abs_z:
        face = 'posy' if y > 0 else 'negy'
        max_axis = abs_y
        u = x / y
        v = (z if y > 0 else -z) / y
    else: 
        face = 'posz' if z > 0 else 'negz'
        max_axis = abs_z
        u = (x if z > 0 else -x) / z
        v = -y / z

    u = 0.5 * (u + 1)
    v = 0.5 * (v + 1)
    return face, u, v


class Fragment:
    def __init__(self, x : int, y : int, depth : float, interpolated_data ):
        self.x = x
        self.y = y
        self.depth = depth
        self.interpolated_data = interpolated_data
        self.output = []

def edgeSide(p, v0, v1) : 
    return (p[0]-v0[0])*(v1[1]-v0[1]) - (p[1]-v0[1])*(v1[0]-v0[0])

class GraphicPipeline:
    def __init__ (self, width, height):
        self.width = width
        self.height = height
        self.image = np.zeros((height, width, 3))
        self.depthBuffer = np.ones((height, width))


    def VertexShader(self, vertex, data):
        outputVertex = np.zeros((15)) 
        
        x = vertex[0]
        y = vertex[1]
        z = vertex[2]
        w = 1.0

        vec = np.array([[x],[y],[z],[w]])
        vec = np.matmul(data['projMatrix'], np.matmul(data['viewMatrix'], vec))

        clip_w = vec[3]
        
        outputVertex[0] = vec[0]/vec[3]
        outputVertex[1] = vec[1]/vec[3]
        outputVertex[2] = vec[2]/vec[3]
        outputVertex[3] = clip_w 

        outputVertex[4] = vertex[3]
        outputVertex[5] = vertex[4]
        outputVertex[6] = vertex[5]

        outputVertex[7] = data['cameraPosition'][0] - vertex[0]
        outputVertex[8] = data['cameraPosition'][1] - vertex[1]
        outputVertex[9] = data['cameraPosition'][2] - vertex[2]

        outputVertex[10] = data['lightPosition'][0] - vertex[0]
        outputVertex[11] = data['lightPosition'][1] - vertex[1]
        outputVertex[12] = data['lightPosition'][2] - vertex[2]

        outputVertex[13] = vertex[6]
        outputVertex[14] = vertex[7]

        return outputVertex

    def Rasterizer(self, v0, v1, v2):
        fragments = []

        area = edgeSide(v0,v1,v2)
        if area < 0:
            return fragments
    
        w0, w1, w2 = v0[3], v1[3], v2[3]
       
        #AABB computation 
        v0_image = np.array([(v0[0]+1.0)/2.0 * self.width, ((v0[1]+1.0)/2.0) * self.height])
        v1_image = np.array([(v1[0]+1.0)/2.0 * self.width, ((v1[1]+1.0)/2.0) * self.height])
        v2_image = np.array([(v2[0]+1.0)/2.0 * self.width, ((v2[1]+1.0)/2.0) * self.height])

        A = np.min(np.array([v0_image,v1_image,v2_image]), axis=0)
        B = np.max(np.array([v0_image,v1_image,v2_image]), axis=0)
        A = np.max(np.array([A, [0.0,0.0]]), axis=0)
        B = np.min(np.array([B, [self.width-1,self.height-1]]), axis=0)
        A = A.astype(int)
        B = B.astype(int) + 1

        for j in range(A[1], B[1]):
            for i in range(A[0], B[0]):
                x = (i+0.5)/self.width * 2.0 - 1 
                y = (j+0.5)/self.height * 2.0 - 1
                p = np.array([x,y])
                
                area0 = edgeSide(p,v0,v1)
                area1 = edgeSide(p,v1,v2)
                area2 = edgeSide(p,v2,v0)

                if (area0 >= 0 and area1 >= 0 and area2 >= 0):
                    lambda0 = area1/area
                    lambda1 = area2/area
                    lambda2 = area0/area
                    
                    z = lambda0 * v0[2] + lambda1 * v1[2] + lambda2 * v2[2]
                    
                    one_over_w = lambda0/w0 + lambda1/w1 + lambda2/w2
                    w = 1.0 / one_over_w
                    
                    l = v0.shape[0]
                    #interpolating
                    interpolated_data = (lambda0*v0[4:l]/w0 + lambda1*v1[4:l]/w1 + lambda2*v2[4:l]/w2) * w
                    fragments.append(Fragment(i,j,z, interpolated_data))
                    
        return fragments
        
    def fragmentShader(self, fragment, data):
        # Extract and normalize normal
        N = fragment.interpolated_data[0:3]
        N = N / np.linalg.norm(N)
        
        # Extract and normalize view vector 
        V = fragment.interpolated_data[3:6]
        V = V / np.linalg.norm(V)
        
        L = fragment.interpolated_data[6:9]
        L = L/np.linalg.norm(L)

        R = 2 * np.dot(L,N) * N  -L

        # Calculate reflection vector
        Re = 2 * np.dot(N, V) * N - V
        Re = Re / np.linalg.norm(Re)

        ambient = 1.0
        diffuse = max(np.dot(N,L),0)
        specular = np.power(max(np.dot(R,V),0.0),64)

        ka = 0.1
        kd = 0.9
        ks = 0.3

        phong = ka * ambient + kd * diffuse + ks * specular
        phong = np.ceil(phong*4 +1 )/6.0

        # Convert to cubemap coordinates
        face, u, v = convert_to_cubemap_coords(Re)
        
        # Sample environment map
        reflection_color = sample(data['environment'][face], u, v)
  
        fragment.output = reflection_color


    def renderSkybox(self, data):

        view = data['viewMatrix'].copy()
        proj = data['projMatrix']

        # Remove translation 
        view[0:3, 3] = 0

        # Invert matrices
        inv_proj = np.linalg.inv(proj)
        inv_view = np.linalg.inv(view)

        for j in range(self.height):
            for i in range(self.width):
                # NDC coordinates 
                x = (i + 0.5) / self.width * 2.0 - 1.0
                y = (j + 0.5) / self.height * 2.0 - 1.0 
                z = 1.0  # Far plane
                ndc = np.array([x, y, z, 1.0])

                #NDC to World direction
                clip = np.dot(inv_proj, ndc)
                clip /= clip[3]  
                world_dir = np.dot(inv_view, clip)[:3]
                world_dir = world_dir / np.linalg.norm(world_dir)  

                # Sample cubemap
                face, u, v = convert_to_cubemap_coords(world_dir)
                self.image[j, i] = sample(data['environment'][face], u, v)
                # Skybox is behind everything


    def draw(self, vertices, triangles, data):
        # Draw background
        self.renderSkybox(data)

        #Calling vertex shader
        self.newVertices = np.zeros((vertices.shape[0], 15))

        for i in range(vertices.shape[0]) :
            self.newVertices[i] = self.VertexShader(vertices[i],data)
        
        fragments = []

        #Calling Rasterizer
        for i in triangles :
            fragments.extend(self.Rasterizer(self.newVertices[i[0]], self.newVertices[i[1]], self.newVertices[i[2]]))
        
        for f in fragments:
            self.fragmentShader(f,data)
            #depth test
            if self.depthBuffer[f.y][f.x] > f.depth : 
                self.depthBuffer[f.y][f.x] = f.depth
                
                self.image[f.y][f.x] = f.output
                
            

