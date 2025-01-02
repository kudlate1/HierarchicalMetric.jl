using Random, Distances, Flux
using Plots
using Test

using Mill
using HSTreeDistance

using JsonGrinder
using JSON3

include("../src/triplet-loss.jl")
include("../src/dataloading.jl")


# X = Float64.([1 3 5 7 9 2 4 6 8 10; 2 4 5 2 5 3 3 5 1 9])
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

X, y = load("julia/data/mutagenesis.json")
distances = pairwiseDistance(X)
heatmap(distances, aspect_ratio = 1)

function train(method::SelectingTripletMethod; λ=0.1, max_iter=200)

    #X, y = load("julia/data/mutagenesis.json")
    metric = reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) 
    # For initialization of weights as 1, use weight_sampler=ones 
    # or more precisely weight_sampler=x -> 0.54 * ones(x)
    # softplus(x) = log(exp(x)+1) ---> 1 = softplus(x) = log(exp(x)+1) ---> x = log(exp(1) - 1) = 0.541324...
    # Random initialization of weights from 𝒩(0,1) is as follows ....  weight_sampler=randn 
    # weights when used in metric are transformed by weight_transform .... softplus(w), where w ∼ 𝒩(0,1)

    ps = Flux.params(metric)
    opt = Adam(λ)

    for iter in 1:max_iter

        anchor, pos, neg = selectTriplet(method, distances, X, y, metric)
        loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric), ps)
        Flux.update!(opt, ps, grad)

        #println("Iteration $iter, loss $loss, params = $ps")
    end

    return ps
end

#-----------------------------------------------------------------------------------------------------------

ps = train(SelectHard())
