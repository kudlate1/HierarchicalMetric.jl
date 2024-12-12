using Random, Distances, Flux
using Plots

using Mill
using HSTreeDistance

using JsonGrinder
using JSON3

include("../src/triplet-loss.jl")
include("../src/dataloading.jl")


# X = Float64.([1 3 5 7 9 2 4 6 8 10; 2 2 2 2 2 3 3 3 3 3])
# y = [1 1 1 1 1 0 0 0 0 0]

# PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
# product_nodes = [PN[i] for i in 1:10]

# pn = product_nodes[1]
# pn.data.x.data |> only

# lx = LeafMetric(Pairwise_SqEuclidean, "con", "x")
# ly = LeafMetric(Pairwise_SqEuclidean, "con", "y")
# PM = ProductMetric((x = lx, y = ly), SqWeightedProductMetric, WeightStruct((x = 1f0, y = 1f0), softplus), "name")
#
# PM2 = reflectmetric(product_nodes[1])
# only(PM2(product_nodes[1], product_nodes[2])) |> typeof

function plotData(points, labels)

    x_coords = vec([only(point.data.x.data) for point in points])
    y_coords = vec([only(point.data.y.data) for point in points])

    println(x_coords)
    println(y_coords)

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

function train(method::SelectingTripletMethod; λ = 0.01, max_iter = 200)

    X, y = load("julia/data/mutagenesis.json")
    metric = reflectmetric(X[1], weight_transform=softplus)
    ps = Flux.params(metric)
    opt = Descent(λ)

    for iter in 1:max_iter

        triplet = selectTriplet(method, X, y, metric)
        (triplet === nothing) && break 

        anchor, pos, neg = triplet
        (pos === nothing || neg === nothing) && continue

        loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric), ps)
        Flux.update!(opt, ps, grad)

        println("Iteration $iter, loss $loss, params = $ps")
    end

    return w
end

#-----------------------------------------------------------------------------------------------------------

# X, y = load("julia/data/mutagenesis.json")

# metric = reflectmetric(X[1])
# metric(X[1], X[2])
# Flux.params(metric)

# original dataset
plotData(product_nodes, y)
w = train(SelectHard())
