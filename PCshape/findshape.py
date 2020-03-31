import numpy as np
from shapely.ops import cascaded_union, polygonize, unary_union
from scipy.spatial import Delaunay
import shapely.geometry as geometry
import math

# Using alpha-shape
def alpha_shape(coords, alpha):
    """
    Compute the alpha shape (concave hull) of a set
    of points.
    #Inputs
        coords: point coordinates. numpy.array with shape=(n, 2). n is number of points
        alpha: factor to select triangle to connect graph
    #Return
        The biggest polygon object that covers the area
    """
    if len(coords) < 4:
        # Find a triangle, no need to compute alpha shape
        return geometry.MultiPoint(list(coords)).convex_hull
    
    def add_edge(edges, edge_points, coords, i, j):
        """
        Add a line between the i-th and j-th points, if not added yet
        """
        if (i, j) in edges or (j, i) in edges:
            # already added
            return
        edges.add( (i, j) )
        edge_points.append(coords[ [i, j] ])
            
    # coords = points
    tri = Delaunay(coords) # Get Delaunary triangles
    edges = set()
    edge_points = [] # list to add edge points
    
    # ia, ib, ic = indices of corner points of the triangle
    for ia, ib, ic in tri.vertices:
        pa = coords[ia]
        pb = coords[ib]
        pc = coords[ic]
        # Lengths of sides of triangle
        a = math.sqrt((pa[0]-pb[0])**2 + (pa[1]-pb[1])**2)
        b = math.sqrt((pb[0]-pc[0])**2 + (pb[1]-pc[1])**2)
        c = math.sqrt((pc[0]-pa[0])**2 + (pc[1]-pa[1])**2)
        # Semiperimeter of triangle
        s = (a + b + c)/2.0
        # Area of triangle by Heron's formula
        area = math.sqrt(s*(s-a)*(s-b)*(s-c))
        circum_r = a*b*c/(4.0*area)

        if circum_r < alpha:
            add_edge(edges, edge_points, coords, ia, ib)
            add_edge(edges, edge_points, coords, ib, ic)
            add_edge(edges, edge_points, coords, ic, ia)
            
    m = geometry.MultiLineString(edge_points) 
    triangles = list(polygonize(m)) # MultiLineString to mutiple polygons(triangles)
    multipolygons = cascaded_union(triangles) # Union triangles
    
    # Sort all polygons by their areas, find the biggest polygon
    if isinstance(multipolygons, geometry.multipolygon.MultiPolygon):
        biggest_polygon = sorted(list(multipolygons), key=lambda x : -x.area)[0] # find biggest cover
    else:
        biggest_polygon = multipolygons
        
    return biggest_polygon



# Using dbscan to extract shapes
def find_boundary(dist, data_index, eps, MinPts):
    """
     point type: 1 for core point, 2 for boundary point, 0 for noise point
    """
    point_type = {}
    for i in data_index: point_type[i] = 0
        
    
    for i in data_index:
        if len(dist[i]) >= MinPts-1 and dist[i][MinPts-2][0]<eps: # i is core point
            point_type[i] = 1
            for p in dist[i]: # check core point's neighbor
                if p[0] < eps: # potential boundary point
                    if point_type[p[1]] != 1: # boundary point cannot override core point
                        point_type[p[1]] = 2
                else: 
                    break
        
    return point_type



