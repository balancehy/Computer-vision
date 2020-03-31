from matplotlib import pyplot as plt
import numpy as np

def plot_boundary(pt_bound, data, index, line_func=dict()):
    """
    Helper function to plot interest point set, boundary points and line segments.
    """
    plt.figure(figsize=[8,8])
    plt.scatter(data[index, 1], data[index, 0], label = "Inner")
    plt.scatter(pt_bound[:, 0], pt_bound[:, 1], label = "Boundary")

#     plt.plot( pt_bound[:, 0],pt_bound[:, 1], 'r--', lw=2)
    
    for ends, func in line_func.items():
        if func:
            a, b = func
            
            x1 = pt_bound[ends[0], 0]
            x2 = pt_bound[ends[1], 0]
            # Corner case, two points have same x
            if x1 == x2:
                x2 += x2*0.01
            y1 = a*x1 + b
            y2 = a*x2 + b
            plt.plot([x1, x2], [y1, y2], 'r--', lw=2)
    
    plt.legend(loc=1, ncol=3)
    plt.xlim([100, 300])
    plt.ylim([100, 300]),
    plt.grid()
    plt.show()

def plot_bd_dbscan(point_type, data):
    """
    Help function for plotting boundary using DBSCAN model
    """
    pt = np.array(list(point_type.items()))
    lb = {1: "core point", 2: "boundary point", 0: "noise point"}
    
    for tp in np.unique(pt[:, 1]):

        idx = np.where(pt[:, 1] == tp)[0]
        plt.scatter(data[:, 1][pt[idx, 0]], data[:, 0][pt[idx, 0]], label = lb[tp])
        if tp == 2:
            res = np.stack([data[pt[idx, 0], 0], data[pt[idx, 0], 1]], axis=1)
    
    plt.legend(loc=1, ncol=3)
    plt.xlim([100, 300])
    plt.ylim([100, 300]),
    plt.grid()
    plt.show()
    
    return res