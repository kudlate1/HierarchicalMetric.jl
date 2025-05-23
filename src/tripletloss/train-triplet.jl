function train_tl_htd(method::TripletSelectionMethod, X, y; λ=0.03, α = 5.0, λₗₐₛₛₒ = 0.1, max_iter=100)
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
    metric = reflectmetric(X[1], weight_sampler=rand, weight_transform=softplus)
    opt = Adam(λ)
    history = []
    old_loss = Inf
    _params = softplus.(Flux.destructure(metric)[1])
    iters = 0
    for iter in 1:max_iter
        iters = iter
        anchor, pos, neg = select_triplet_htd(method, X, y, metric)
        state_tree = Flux.setup(opt, metric)
        loss, grad = Flux.withgradient(metric) do m
            triplet_loss_htd_global(anchor, pos, neg, m; α = α, λₗₐₛₛₒ = λₗₐₛₛₒ, weight_transform=softplus)
        end
        Flux.update!(state_tree, metric, grad[1])
        _params = softplus.(Flux.destructure(metric)[1])
        push!(history, _params)
        println("Iteration $iter, loss $loss, triplet (a, p, n) = ($anchor, $pos, $neg), params = $_params")
    end
    return _params, history, iters
end
