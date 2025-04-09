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

function create_product_nodes(data)

    PN = ProductNode((x = Array(data[1, :]'), y  = Array(data[2, :]')))
    X = [PN[i] for i in 1:10]
    return X
end

function visualise_distances(distances)
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
