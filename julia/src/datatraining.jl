using Random, Distances, Flux

using Mill
using HSTreeDistance

include("../src/triplet-loss.jl")
include("../src/dataloading.jl")


function train(method::TripletSelectionMethod, X, y, distances; λ=0.1, max_iter=50)

    """
    A triplet-based training method using the Flux.jl library

    Params:
    method (SelectingTripletMethod): the method of triplet selection (hard or random)
    λ (float): learning rate
    max_iter (int): the numberof training iterations

    Return:
    (Zygote.Params{Zygote.Buffer{Any, Vector{Any}}}): values of the trained parameters
    (Vector{Any}): an array of parameters (history of learning)

    """

    metric = reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) 
    # For initialization of weights as 1, use weight_sampler=ones 
    # or more precisely weight_sampler=x -> 0.54 * ones(x)
    # softplus(x) = log(exp(x)+1) ---> 1 = softplus(x) = log(exp(x)+1) ---> x = log(exp(1) - 1) = 0.541324...
    # Random initialization of weights from 𝒩(0,1) is as follows ....  weight_sampler=randn 
    # weights when used in metric are transformed by weight_transform .... softplus(w), where w ∼ 𝒩(0,1)

    metric |> typeof
    ps = Flux.params(metric)
    opt = Adam(λ)
    history = []
    for iter in 1:max_iter

        anchor, pos, neg = selectTriplet(method, distances, X, y, metric)
        loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; weight_transform=softplus), ps)
        Flux.update!(opt, ps, grad)
        push!(history, reduce(vcat, ps))
        println("Iteration $iter, loss $loss, history = $(history[iter]), params = $ps)")
    end

    return ps, history
end
