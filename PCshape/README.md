# Point cloud shape finding and boundary extracting

* Use DBscan to cluster the original data, use the biggest cluster for shape finding
* Use two approaches to find shape(concave hull) of the cluster
    - Implement alpha-shape algorithm to find the concave hull of the cluster.
    - Use boundary points in DBscan model to represent cluster shape. This method is either find a tight boundary and treat many sparse area as noise, or it cannot find a complete boundary, because of the limitation of density-based algorithm. 
* Use split-merge algorithm to extract line segments based on the boundary points