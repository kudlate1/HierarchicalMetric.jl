using HierarchicalMetric

function plot_points(X, centroids)

    data = hcat(X, centroids)
    
    points = Int64.(ones(1, length(X[1, :])))
    centers = Int64.(ones(1, length(centroids[1, :])))

    #labels = hcat(points, centers)
    labels = vcat(zeros(size(X, 2)), ones(size(centroids, 2))) # just to show X and centroids with different colors
    @info labels

    plotData(data, labels)
end

"""
EXAMPLE OF LAC EXECUTION:

X = [ 2.5  1.8 -2.1  3.9 -2.5  2.1 ;
      3.2  2.9 -3.1 -2.0 -2.8 -3.3 ]

centroids = [ -2.0  2.0 3.1 ;
              -2.0  1.2 -1.1 ]

k = 3
d = 2

plot_points(X, centroids)
new_centroids = LAC(X, k, Lw())
plotPoints(X, new_centroids)

"""