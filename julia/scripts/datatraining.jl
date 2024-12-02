using Random, Distances, Flux
using Plots
using Mill
using HSTreeDistance

include("../src/triplet-loss.jl")


X = Float64.([1 3 5 7 9 2 4 6 8 10; 2 2 2 2 2 3 3 3 3 3])
y = [1 1 1 1 1 0 0 0 0 0]

PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
product_nodes = [PN[i] for i in 1:10]

# pn = product_nodes[1]
# pn.data.x.data |> only

# lx = LeafMetric(Pairwise_SqEuclidean, "con", "x")
# ly = LeafMetric(Pairwise_SqEuclidean, "con", "y")
# PM = ProductMetric((x = lx, y = ly), SqWeightedProductMetric, WeightStruct((x = 1f0, y = 1f0), softplus), "name")
#
# PM2 = reflectmetric(product_nodes[1])

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

function train(method::SelectingTripletMethod, product_nodes; λ = 0.01, epochs = 200)

    metric = reflectmetric(product_nodes[1])
    w = metric.weights.values
    ps = Flux.params(w)
    opt = Descent(λ)

    for epoch in 1:epochs

        triplet = selectTriplet(method, product_nodes, y, metric)
        (triplet == nothing) && break 

        anchor, pos, neg = triplet
        (pos == nothing || neg == nothing) && continue

        loss, grad = Flux.withgradient(() -> triplet_loss(anchor, pos, neg, metric), ps)
        Flux.update!(opt, ps, grad)

        println("Epoch $epoch, loss $loss, [w1,w2] = $w")
    end

    return w
end

#-----------------------------------------------------------------------------------------------------------

# original dataset
plotData(product_nodes, y)

w = train(SelectHard(), product_nodes)
X_modified = deepcopy(X)

for i in X
    for j in 1:length(X)
        i[j] = i[j] * w[j]
    end
end

# dataset after applying trained parameters w1, w2
plotData(X_modified, y)
