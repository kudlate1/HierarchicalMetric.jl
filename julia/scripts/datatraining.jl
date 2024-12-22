using Random, Distances, Flux
using Plots

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

function train(method::SelectingTripletMethod; λ = 0.01, max_iter = 200)

    X, y = load("julia/data/mutagenesis.json")
    metric = reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) 
    # For initialization of weights as 1, use weight_sampler=ones 
    # or more precisely weight_sampler=x -> 0.54 * ones(x)
    # softplus(x) = log(exp(x)+1) ---> 1 = softplus(x) = log(exp(x)+1) ---> x = log(exp(1) - 1) = 0.541324...
    # Random initialization of weights from 𝒩(0,1) is as follows ....  weight_sampler=randn 
    # weights when used in metric are transformed by weight_transform .... softplus(w), where w ∼ 𝒩(0,1)

    ps = Flux.params(metric)
    opt = Descent(λ)

    for iter in 1:max_iter

        triplet = selectTriplet(method, distances, X, y, metric)
        (triplet === nothing) && break 

        anchor, pos, neg = triplet
        (pos === nothing || neg === nothing) && continue

        loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric), ps)
        Flux.update!(opt, ps, grad)

        println("Iteration $iter, loss $loss, params = $ps")
    end

    return ps
end

#-----------------------------------------------------------------------------------------------------------

# X, y = load("julia/data/mutagenesis.json")

# metric = reflectmetric(X[1])
# metric(X[1], X[2])
# Flux.params(metric)

# original dataset
# plotData(product_nodes, y)
# w = train(SelectHard())



# test if there any gradients from lasso objective
function test_lasso(method::SelectingTripletMethod; λ = 0.01)
    X, y = load("julia/data/mutagenesis.json")
    triplet = selectTriplet(method, X[1:10], y[1:10], metric) # just on small batch
    anchor, pos, neg = triplet
    
    @testset "Testing gradients of triplet loss with lasso" verbose=true begin
        @testset "testing contribution of only lasso λₗₐₛₛₒ = 1.0" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            ps = Flux.params(metric)
            opt = Descent(λ)
            loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; α=-Inf, λₗₐₛₛₒ=1f0, weight_transform=identity), ps)
            # if α= -Inf or number small enaugh, the triplet loss contribution to loss function will be zero
            # And loss will be equal to regularization term from lasso


            @test loss ≈ sqrt(sum(ones(13))) # sqrt(13)
            @test reduce(vcat, ps) == ones(13)
            Flux.update!(opt, ps, grad) # check if parameters after output are those what we wished for
            @test reduce(vcat, ps) ≈ ones(13) .- λ * sqrt(13)/13
        
        end

        @testset "testing contribution of lasso λₗₐₛₛₒ = 0.5" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            ps = Flux.params(metric)
            opt = Descent(λ)
            loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; α=-Inf, λₗₐₛₛₒ=0.5, weight_transform=identity), ps)

            @test loss ≈ 0.5*sqrt(sum(ones(13))) # sqrt(13)
            @test reduce(vcat, ps) == ones(13)
            Flux.update!(opt, ps, grad) # check if parameters after output are those what we wished for
            @test reduce(vcat, ps) ≈ ones(13) .- λ * sqrt(13)/13*0.5
        
        end
        
        @testset "testing contribution of tripletloss together with lasso λₗₐₛₛₒ = 1.0" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            ps = Flux.params(metric)
            opt = Descent(λ)
            loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; α=1, λₗₐₛₛₒ=1f0, weight_transform=identity), ps)

            @test !(loss ≈ 1f0 * sqrt(sum(ones(13)))) # sqrt(13)
            @test reduce(vcat, ps) == ones(13)
            Flux.update!(opt, ps, grad) # check if parameters after output are those what we wished for
            @test !(reduce(vcat, ps) ≈ ones(13) .- λ * sqrt(13)/13)
        
        end
    end
end

test_lasso(SelectHard());

ps = train(SelectHard())
