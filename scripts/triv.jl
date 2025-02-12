using HierarchicalMetric

"""
Example of usage of functions 

Values in the ProductNode access:
pn = product_nodes[1]
pn.data.x.data |> only  # elementary data type (int, str,...)

Creating the metric using ProductMetric:
lx = LeafMetric(Pairwise_SqEuclidean, "con", "x")
ly = LeafMetric(Pairwise_SqEuclidean, "con", "y")
PM = ProductMetric((x = lx, y = ly), SqWeightedProductMetric, WeightStruct((x = 1f0, y = 1f0), softplus), "name")

Creating the metric using reflectmetric (equivivalent):
PM2 = reflectmetric(product_nodes[1])

Computing the distance of two ProductNodes using metric (PM/PM2):
only(PM2(product_nodes[1], product_nodes[2]))

"""

function trainTrivial()

    function plotData(points, labels; w = [1.0, 1.0])

        x_coords = vec(w[1] * [only(point.data.x.data) for point in points])
        y_coords = vec(w[2] * [only(point.data.y.data) for point in points])
    
        colors = [label == 1 ? RGB(0.2, 0.2, 0.8) : RGB(0.4, 0.1, 0.2) for label in vec(labels)]
    
        scatter(
            x_coords,
            y_coords,
            color = colors,
            marker = (10, :circle),
            xlabel = "x",
            ylabel = "y",
            background_color = RGB(0.2, 0.2, 0.2),
            legend = false
        )
    end

    # trivial artificial dataset and its labels 
    data = Float64.([1 3 5 7 9 2 4 6 8 10; 2 2 2 2 2 3 3 3 3 3])
    y = [1 1 1 1 1 0 0 0 0 0]

    # making the ProductNodes from basic matrix notation
    PN = ProductNode((x = Array(data[1, :]'), y  = Array(data[2, :]')))
    X = [PN[i] for i in 1:10]
    distances = pairwiseDistance(X)

    # ploting the original data as graph and heatmap
    plotData(X, y)
    heatmap(distances, aspect_ratio = 1)

    # training the parameters and ploting the result
    ps, h = train(SelectHard(), X, y, distances)
    plotData(X, y; w = softplus.(ps)[1])

end