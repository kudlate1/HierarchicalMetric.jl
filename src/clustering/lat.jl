function LAT_vec(init::InitCenters, X::Matrix, k::Int; h::Real=2.0, λ::Real=0.0008, max_iter::Int=50)
    d, n = size(X)

    # 1. initialization of parameters
    idxs = rand(1:n, k) #init_centers(init, X, k)
    c = X[:, idxs]
    γ = zeros(n, k)
    W = [Matrix(1.0I, d, d) for _ in 1:k]
    π = fill(1/k, k)
    y = zeros(Int64, n)
    opt = Adam(λ)
    history = []
    loglike_init = -Inf
    iters = 0

    losses = []

    for iter in 1:max_iter
        iters = iter
        last_weights = deepcopy(W)
        # 2. E-step: compute responsibilities and cluster points
        compute_responsibilities_lat(X, c, W, π, γ, h)
        y = classify(γ)

        # plot_distributions_2d(X, c, [w * w' for w in W], γ)
        # savefig("images/distr_phase_$iter")

        # 3. M-step: update centroids and mixture coefficients
        N = sum(γ, dims=1)[:]
        π = N / n
        idxs = update_centers_lat(X, idxs, y, W)
        c = X[:, idxs]
        # 4. triplet loss: update weights W
        for j in 1:10
            anchor, pos, neg, y_p, y_n = select_triplet_lat_vec(X, c, y, W)
            (y_n == 0) && return c, idxs, ones(Int64, n), [w * w' for w in W], probs, history, losses, iters
            state_tree = Flux.setup(opt, W)
            loss, grad = Flux.withgradient(W) do w
                triplet_loss_vec(anchor, pos, neg, y_p, y_n, w; α = 0.5, λₗₐₛₛₒ = 0.01)
            end
            Flux.update!(state_tree, W, grad[1])
            push!(losses, loss)
            #println("Iteration $iter.$j, loss $loss, centroids $c")
        end
        push!(history, W)
        println("Iteration $iter, centers $c, params = $W")
        # 5. Convergence check
        weight_diff = sum(sum((W[i] .- last_weights[i]).^2) for i in 1:k)
        (weight_diff < 1e-4) && break
    end
    weights = [w * w' for w in W]
    return c, idxs, y, weights, γ, history, losses, iters
end

function LAT_htd(init::InitCenters, X, k::Int; h::Real=2.0, λ::Real=0.00008, max_iter::Int=20)
    n = length(X)

    # 1. initialization of parameters
    idxs = rand(1:n, k)
    c = X[idxs]
    γ = zeros(n, k)
    W = [reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) for _ in 1:k]
    π = fill(1/k, k)
    y = zeros(Int64, n)
    opt = Adam(λ)
    history = []
    iters = 0

    for iter in 1:max_iter
        iters = iter
        old_p = deepcopy(W)
        old_params = [softplus.(Flux.destructure(w)[1] for w in old_p)]
        _old_params = reduce(vcat, reduce(vcat, old_params))

        # 2. E-step: compute responsibilities and cluster points
        compute_responsibilities_htd(X, c, W, π, γ, h)
        y = classify(γ)
        y_1 = sum([1 for i in 1:n if y[i] == 1])
        y_2 = sum([1 for i in 1:n if y[i] == 2])
        @show y, y_1, y_2

        # 3. M-step: update centroids and mixture coeffs
        N = sum(γ, dims=1)[:]
        idxs = update_centers_lat_htd(X, idxs, y, W)  # k-medoids
        c = X[idxs]
        π = N / n

        # 4. triplet loss: update weights W
        for j in 1:10
            anchor, pos, neg, y_p, y_n = select_triplet_lat_htd(X, c, y, W)
            (y_n == 0) && return c, idxs, ones(Int64, n), W, γ, history, iters
            state_tree = Flux.setup(opt, W)
            loss, grad = Flux.withgradient(W) do w
                triplet_loss_htd(anchor, pos, neg, y_p, y_n, w; α = 0.5, λₗₐₛₛₒ = 0.01, weight_transform=softplus)
            end
            println("Iter $iter.$j, loss = $loss")
            Flux.update!(state_tree, W, grad[1]) 
        end
        p = [softplus.(Flux.destructure(w)[1] for w in W)]
        _params = reduce(vcat, reduce(vcat, p))
        println("Iter $iter, w1 = $(p[1][1]), w2 = $(p[1][2])")
        push!(history, _params) 

        @show old_params
        @show _params

        # 5. Convergence check
        metric_diff = sum(sum(_params[i] .- _old_params[i]).^2 for i in 1:k)
        (metric_diff < 1e-4) && break
    end
    p = [softplus.(Flux.destructure(w)[1] for w in W)][1]
    return c, idxs, y, p, γ, history, iters
end
