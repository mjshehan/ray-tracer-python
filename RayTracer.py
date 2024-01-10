""""
CSC 305
Assignment 3 - Backward Ray Tracer
Michael Shehan
V00203133
Submitted December 4, 2023
"""

import sys
import numpy as np

class Scene:
    """
    Scene class, contains camera, near plane, list of objects, background, lights, and resolution information 
    """    
    def __init__(self, camera, near, left, right, bottom, top, res_x, res_y, objects, background, lights, ambient):
        self.camera = camera
        self.near = near
        self.left = left
        self.right = right
        self.bottom = bottom
        self.top = top
        self.nCols = res_x
        self.nRows = res_y
        self.objects = objects
        self.background = background
        self.lights = lights
        self.ambient = ambient
    def __str__(self):
        return f"Scene: {self.near}, {self.left}, {self.right}, {self.bottom}, {self.top}, {self.res_x}, {self.res_y}, {self.objects}, {self.background}, {self.lights}, {self.ambient}"

class Sphere:
    """"
    Sphere class for sphere objects in the scene
    Also includes the transformation matrix and inverse transpose matricies for the sphere
    """
    def __init__(self, name, pos_x, pos_y, pos_z, scl_x, scl_y, scl_z, ir, ig, ib, k_a, k_d, k_s, k_r, n_s):  
        if(scl_x == 0):
            scl_x = 1
        if(scl_y == 0):
            scl_y = 1
        if(scl_z == 0):
            scl_z = 1    
        self.name = name
        self.pos = np.array([pos_x, pos_y, pos_z])
        self.scale = np.array([scl_x, scl_y, scl_z])
        self.radius = scl_z
        self.ir = ir
        self.ig = ig
        self.ib = ib
        self.k_a = k_a
        self.k_d = k_d
        self.k_s = k_s
        self.k_r = k_r
        self.n_s = n_s        
        self.transfrom_matrix = np.array([[scl_x, 0, 0, pos_x], [0, scl_y, 0, pos_y], [0, 0, scl_z, pos_z], [0, 0, 0, 1]])
        self.inverse_transform = np.linalg.inv(self.transfrom_matrix)
        inverse_transpose = (np.linalg.inv(self.transfrom_matrix)).T
        self.inverse_transpose = inverse_transpose[:3, :3]  
    def __str__(self):
        return f"Sphere: {self.name}, {self.pos}, {self.radius}, {self.ir}, {self.ig}, {self.ib}"    

class Ray:
    """Ray class for ray objects in the scene"""
    def __init__(self, origin, direction):
        self.origin = origin
        self.direction = direction
        self.depth = np.linalg.norm(self.direction)
        self.normalized = self.direction / self.depth
    def __str__(self):
        return f"Ray: {self.origin}, {self.direction}"

class Light:
    """Light class for light objects in the scene"""
    def __init__(self, name, pos_x, pos_y, pos_z, ir, ig, ib):
        self.name = name
        self.pos = np.array([pos_x, pos_y, pos_z])
        self.ir = ir
        self.ig = ig
        self.ib = ib

class Renderer:
    """"
    Renderer Class 
    methods: render, ray_trace, find_closest, intersects, get_colour
    """
    def render(self, scene):
        """"
        Method calls ray_trace() for each pixel in the scene
        appends pixel colours to a list, to be written to ppm file
        Returns: list of pixels
        """
        width = scene.nCols
        height = scene.nRows
        aspect_ratio = width / height
        x0 = scene.left
        x1 = scene.right
        xstep = (x1 - x0) / (width - 1)
        y0 = scene.bottom / aspect_ratio
        y1 = scene.top / aspect_ratio
        ystep = (y1 - y0) / (height - 1)
        camera = scene.camera
        pixels = []
    
        for j in range(height):
            y = y0 + j * ystep
            for i in range(width):
                x = x0 + i * xstep
                ray = Ray(camera, np.array([x, -y, -scene.near]))
                ray_normalized = Ray(camera, ray.normalized)
                pixels.append(self.ray_trace(ray_normalized, scene, depth = 0))
        return pixels
   
    def ray_trace(self, ray, scene, depth):
        """
        Class method, called recursively to trace rays
        calls find_closest(), get_colour(), methods
        if a ray from the eye hits nothing, returns background colour,
        if a reflected ray hits nothing, returns black 
        """
        depth = depth
        colour = np.array(scene.background)
        dist_hit, obj_hit, canon_normal = self.find_closest(ray, scene)
        
        if obj_hit is None:
            if ray.origin is not scene.camera:
                return np.array([0.0,0.0,0.0])
            else:
                return colour
        hit_pos = ray.origin + ray.direction * dist_hit
     
        if hit_pos[2] > -scene.near and obj_hit.name != "innie": # special cases for objects that intersect the near plane
            return colour
        else:
            return_colour = self.get_colour(obj_hit, canon_normal, hit_pos, scene, depth) 
            return_colour[0] = min(return_colour[0], 1)
            return_colour[1] = min(return_colour[1], 1)
            return_colour[2] = min(return_colour[2], 1)
            return return_colour
    
    def intersects(self, ray, sphere):
        """
        Intersects method called by find_closest()
        Solves quadratic equation for t to test for intersection
        if t is positive, returns the minimum t value as distance to the intersection
        """
        # Calculate the vector from the ray's origin to the sphere's center
        S = ray.origin - sphere.pos # origin of ray to center of sphere
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(S, ray.direction)
        c = np.dot(S, S) - 1
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        elif discriminant == 0:
            t = -b / ( 2*a)  # should this be divide by /2a
            return 0
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2*a) # shoudl this be divide by /2a
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            if t1 > 0 and t2 > 0:
                return min (t1, t2) 
            elif t1 > 0:
                return t1
            elif t2 > 0 :
                return t2
            else:
                return None
            # return the distance to the intersection, not the intersection point itself
    
    def intersects_near_plane(self, ray, sphere):
        """
        Class method for special case of objects that intersect the viewing plane
        """
        #print("we are in the NEAR PLANE!!!!!!!!!!")
        
        # formula from ray tracing slide deck
        S = ray.origin - sphere.pos # origin of ray to center of sphere
        a = np.dot(ray.direction, ray.direction)
        b = 2.0 * np.dot(S, ray.direction)
        c = np.dot(S, S) - 1
        discriminant = b**2 - 4*a*c
        
        if discriminant < 0:
            return None
        elif discriminant == 0:
            t = -b / ( 2*a)  
            return t
        else:
            t1 = (-b + np.sqrt(discriminant)) / (2*a) 
            t2 = (-b - np.sqrt(discriminant)) / (2*a)
            if t1 > 0 and t2 > 0:
                t = max(t1, t2)
                return(t)
            elif t1 > 0:
                return t1
            elif t2 > 0 :
                return t2
            else:
                return None
    
    def find_closest(self, ray, scene):
        """
        Class method to find the closest object that intersects with the ray
        uses the intersects() method to find the distance to the intersection
        Returns: distance to the closest object, the object itself, and the normal of the object as canonical sphere
        """
        dist_min = None
        obj_hit = None
        obj_normal = None
        dist_min = 0
        dist = None
        for obj in scene.objects:
            canonical_sphere = Sphere("cannonical", 0,0,0,1,1,1, obj.ir, obj.ig, obj.ib, obj.k_a, obj.k_d, obj.k_s, obj.k_r, obj.n_s)      
            t_ray = transform_ray(ray, obj)
            
            if obj.pos[2] + obj.radius > -scene.near:
                obj.name = "innie"  
                dist = self.intersects_near_plane(t_ray, canonical_sphere) 
                if dist is not None:
                    hit_pos = ray.origin + ray.direction * dist
                    if hit_pos[2] > -scene.near:
                        dist = None
            else:     
                dist = self.intersects(t_ray, canonical_sphere)
            if dist is not None and (obj_hit is None or dist < dist_min):
                    dist_min = dist
                    obj_hit = obj
                    obj_normal = t_ray.origin + t_ray.direction * dist_min  
                    if(obj_hit.name == "innie"):
                        obj_normal = obj_normal * -1
                                       
      
        return (dist_min, obj_hit, obj_normal)

    def get_colour(self, obj_hit, canon_normal, hit_pos, scene, depth):
        """
        Class method to calculate the colour for a particular pixel
        Recursively calls ray_trace() to calculate the colour of reflected rays at depth set at max_depth
        Args: obj_hit: object that was hit, canon_normal: normal of the object as canonical sphere, hit_pos: position of the hit, scene: scene object, depth: depth of the ray
        Returns: colour of the pixel [r,g,b]
        """
        
        max_depth = 3 # max depth of recursion
        pixel_colour = np.array([0.0,0.0,0.0]) # array to return the colour of the pixel
        obj_colour = np.array([obj_hit.ir, obj_hit.ig, obj_hit.ib])
        transformed_normal = obj_hit.inverse_transpose.dot(canon_normal)
        transformed_normal = transformed_normal / np.linalg.norm(transformed_normal)  
        
        
        ### ADS COMPUTATIONS:
        if obj_hit.k_a > 0:
           pixel_colour += obj_hit.k_a * obj_colour * scene.ambient

        #if hit_pos[2] > -scene.near:
        #   print("found a booby at ", hit_pos )
      
        for light in scene.lights:
            light_colour = np.array([light.ir, light.ig, light.ib])
            to_light = Ray(hit_pos, light.pos - hit_pos ) 
            to_light.direction = to_light.normalized  # normalize the direction
            reflection_epsilon = transformed_normal * 0.00001
            to_light.origin += reflection_epsilon

            ###---SHADOW---###
            if self.find_closest(to_light, scene)[1] is not None:
                if obj_hit.name != "innie":
                    continue
                elif obj_hit.name == "innie" and light.pos[2] < -scene.near:
                    return pixel_colour
                else:
                    if light.pos[2] >= -scene.near:
                       transformed_normal = -transformed_normal
            
            ###---DIFUSE---###
            difuse = np.array([0.0,0.0,0.0])
            if obj_hit.k_d > 0:
                transformed_normal = transformed_normal
                difuse = obj_hit.k_d * light_colour * max(transformed_normal.dot(to_light.direction), 0) * obj_colour
            
            ###---SPECULAR---### 
            specular = np.array([0.0,0.0,0.0])
            R = (2 * max(transformed_normal.dot(to_light.direction),0) * transformed_normal)-  to_light.direction 
            R = R / np.linalg.norm(R) 
            if obj_hit.k_s > 0:
                V =  scene.camera - hit_pos 
                V = V/ np.linalg.norm(V)
                R_dot_V = max(R.dot(V), 0)
                specular = light_colour * obj_hit.k_s *  R_dot_V**obj_hit.n_s

            ###---REFLECTION---###
            colour_re = 0
            if depth < max_depth and obj_hit.k_r > 0:
                depth +=1
                reflected_c = R +  reflection_epsilon 
                reflected_ray = Ray(hit_pos, reflected_c)
                ray_trace_colour = np.array(self.ray_trace(reflected_ray, scene, depth))
                colour_re = obj_hit.k_r * ray_trace_colour
    
            pixel_colour += (difuse + specular + colour_re)     
        return pixel_colour

def transform_ray(ray, sphere):
    """
    Transforms a ray from object coordinates to world coordinates using the inverse of the sphere's transformation matrix
    uses formula from ray tracing slide deck 
    """
    ray_origin = np.array([ray.origin[0], ray.origin[1], ray.origin[2], 1])
    ray_direction = np.array([ray.direction[0], ray.direction[1], ray.direction[2], 0])
    Sxyz1 = sphere.inverse_transform.dot(ray_origin) 
    Sxyz = np.array([Sxyz1[0], Sxyz1[1], Sxyz1[2]])
    cxyz0 = sphere.inverse_transform.dot(ray_direction)
    cxyz = np.array([cxyz0[0], cxyz0[1], cxyz0[2]])
    transformed_ray = Ray(Sxyz, cxyz)
    return transformed_ray


def read_file(filename):
    """
    Reads the input file
    Returns: scene object and output filename
    """
    print(f"Reading file: {filename}")
    NEAR = 0
    spheres_in = []
    lights_in = []

    file = open(filename, 'r')
    for line in file:
        words = line.split()
        if not words:
            continue
        if words[0] == 'NEAR':
            NEAR = int(words[1])
        if words[0] == 'LEFT':
            LEFT = int(words[1])
        if words[0] == 'RIGHT':
            RIGHT = int(words[1])
        if words[0] == 'BOTTOM':
            BOTTOM=int(words[1])
        if words[0] == 'TOP':
            TOP = int(words[1])        
        if words[0] == 'RES':
            x = int(words[1])
            y = int(words[2])
            RES = (x, y)
        if words[0] == 'SPHERE':
            name = words[1]
            pos_x = float(words[2])
            pos_y = float(words[3])
            pos_z = float(words[4])
            scl_x = float(words[5])
            scl_y = float(words[6])
            scl_z = float(words[7])
            ir = float(words[8])
            ig = float(words[9])
            ib = float(words[10])
            k_a = float(words[11])
            k_d = float(words[12])
            k_s = float(words[13])
            k_r = float(words[14])
            n_s = float(words[15])
            sphere = Sphere(name, pos_x, pos_y, pos_z, scl_x, scl_y, scl_z, ir, ig, ib, k_a, k_d, k_s, k_r, n_s) 
            spheres_in.append(sphere)
        if words[0] == 'LIGHT':
            lname = words[1]
            lpos_x = float(words[2])
            lpos_y = float(words[3])
            lpos_z = float(words[4])
            lir = float(words[5])
            lig = float(words[6])
            lib = float(words[7])
            light = Light(lname, lpos_x, lpos_y, lpos_z, lir, lig, lib)
            lights_in.append(light)
        if words[0] == 'BACK':
            br = float(words[1])
            bg = float(words[2])
            bb = float(words[3])
            backgroundIn = np.array([br, bg, bb])
        if words[0] == 'AMBIENT':
            ir = float(words[1])
            ig = float(words[2])
            ib = float(words[3])
            ambient = np.array([ir, ig, ib])
        if words[0] == 'OUTPUT':
            output_filename = words[1]
    file.close()
    
    eye = np.array([0,0,0])
    scene = Scene(eye, NEAR, LEFT, RIGHT, BOTTOM, TOP, RES[0], RES[1], objects=spheres_in, background=backgroundIn, lights=lights_in, ambient=ambient)
    return scene, output_filename

def write(pixels, file_out, res_x, res_y):
    """
    Function writes pixels to a ppm file using ppm3
    Args: pixels: list of pixels to be written, file_out: name of file to be written to, res_x: x resolution, res_y: y resolution
    """
    print("Writing file: ", file_out) 
    f = open(f"{file_out}", "w")
    f.write("P3\n")
    f.write(f"{res_x} {res_y}\n")  
    f.write("255\n")
    for i in range(len(pixels)):
        f.write(str(int(pixels[i][0]*255)) + " " + str(int(pixels[i][1]*255)) + " " + str(int(pixels[i][2]*255)) + " ")
    f.close()

def main():
    """
    main function to execute the program
    """

    filename = sys.argv[1]
    scene, file_out = read_file(filename)
    engine = Renderer()
    pixels = engine.render(scene)
    write(pixels, file_out, scene.nCols, scene.nRows)

if __name__ == "__main__":
    main()

