import numpy as np

def fit_line(seg, points):
    """
    Least square method to fit line.
    #Inputs
        seg: the index of current segment endpoints
        points: original point set
    #return
        a list of line parameters [a, b]
    """
    idx = np.arange(seg[0], seg[1]+1)
    x, y = points[idx, 0], points[idx, 1]
    A = np.stack([x, np.ones(x.shape)], axis=1)
    a, b = np.linalg.lstsq(A, y, rcond=None)[0]
    
    return [a, b]

def point2line_dist(pt, line):
    """
    Calculate distance from a point to the given line.
    #Inputs
        pt: 2d point coordiantes
        line: line function parameter
    #Return
        distance
    """
    a, b = line
    up = a*pt[0] - pt[1] + b
    return np.abs(up) / np.sqrt(a**2 + 1)
    

def extract_lines(points, idx_s, idx_t, threshold):
    """
    Split method. Handle splitting at start or end point.
    #inputs
        points: a set of points to extract lines. Must be sorted in clockwise or counter-clockwise order
        idx_s: start index in point set
        idx_t: end index in point set
        threshold: if a point's distance to a fitted line is larger than this threshold, split at this point
        
    #Return: dictionary. Key is index of line segment endpoints. Value is line parameters [a, b ]in line function
        y = a*x + b
    """
    segments = [[idx_s, idx_t]] # [first, last]
    line_func = {}
    
    single_points = []
    while(segments or single_points):
        
        if not segments:
            segments.append([single_points[-1], single_points[0]])
        seg = segments.pop(-1)
        
        # If pop single point, stack it and merge them later
        if seg[1] - seg[0] == 0:
            single_points.append(seg[0])
            continue
        
        line = fit_line(seg, points)
        if seg[1] - seg[0] == 1:
            line_func[tuple(seg)] = line
            # Bridge gap if popped segment is not continuous with the top segment
            if segments and segments[-1][1] == seg[0]-1:
                segments.append([segments[-1][1], seg[0]])
            # Merge single point, append to stack
            if single_points:
                segments.append([seg[1], single_points[0]])
                single_points = []
            continue
        
        # Find most distant point
        split_index = -1 # one bigger than the last index in set
        max_dist = 0.0
        for i in range(seg[0], seg[1]+1):
            pt = points[i]
            dist = point2line_dist(pt, line)
            if dist > max_dist:
                max_dist = dist
                split_index = i

        if split_index != -1 and max_dist > threshold: # can split
            left = [seg[0], split_index]
            right = [split_index, seg[1]]
            # Handle corner case when split point is start or end point. Can trigger discontinous line segments,
            # which will be handled when popping valid segment
            if split_index == seg[0]:
                right = [split_index+1, seg[1]]
            if split_index == seg[1]:
                left = [seg[0], split_index-1]
            
            segments.append(left)
            segments.append(right)
            if right[1] != right[0] and single_points:
                segments.append([right[1], single_points[0]])
                single_points = []
            
        else:
            line_func[tuple(seg)] = line
            # Bridge gap
            if segments and segments[-1][1] == seg[0]-1:
                segments.append([segments[-1][1], seg[0]])
            # Merge single point, append to stack
            if single_points:
                segments.append([seg[1], single_points[0]])
                single_points = []
    
    
    return line_func