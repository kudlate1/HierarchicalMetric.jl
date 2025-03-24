using HierarchicalMetric
using HierarchicalMetric.Mill
using Plots

"""
Example of usage of Mill.jl functions 

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

function plotData(data, labels)

    x_coords = vec([i[1] for i in eachcol(data)])
    y_coords = vec([i[2] for i in eachcol(data)])

    colors = [label == 1 ? RGB(0.2, 0.2, 0.8) : RGB(0.4, 0.1, 0.2) for label in vec(labels)]

    scatter(
        x_coords,
        y_coords,
        color = colors,
        marker = (10, :circle),
        xlabel = "x",
        ylabel = "y",
        background_color = RGB(0.4, 0.4, 0.4),
        xlims=(-0.1, 0.9),
        ylims=(0.2, 0.5),
        legend = false
    )
end

function createProductNodes(data)

    PN = ProductNode((x = Array(data[1, :]'), y  = Array(data[2, :]')))
    X = [PN[i] for i in 1:10]
    return X
end

function visualiseDistances(distances)
    heatmap(distances, aspect_ratio = 1)
end

# function plotProcess(ps, h)

#     p = plot(reduce(hcat, h)', 
#         xlabel="number of iteations", 
#         ylabel="values of the parameters", 
#         title="Parameters learning with lasso regularization"
#     )
#     display(p)

#     return vcat(softplus.(ps)[1])
# end


"""
EXAMPLE OF A TRAINING EXECUTION:

data = Float64[1 3 5 7 9 2 4 6 8 10; 1 1 1 1 1 2 2 2 2 2]
y = [1 1 1 1 1 2 2 2 2 2]

plotData(data, y)
X = createProductNodes(data)

distances = pairwiseDistance(X)
visualiseDistances(distances)

ps, h = train(SelectRandom(), X, y, distances);
plotProcess(ps, h)

"""
