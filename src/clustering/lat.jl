function LAT_vec(init::InitCenters, X::Matrix, k::Int; h::Real=2.0, λ::Real=0.001, max_iter::Int=200)

    d, n = size(X)

    # 1. initialization of parameters
    c = init_centers(init, X, k)
    γ = zeros(n, k)
    W = [Matrix(1.0I, d, d) for _ in 1:k]
    π = fill(1/k, k)
    y = zeros(Int64, n)
    opt = Adam(λ)
    history = []
    loglike_init = -Inf
    iters = 0

    for iter in 1:max_iter

        iters = iter
        last_weights = W
        last_centroids = c

        # 2. E-step: compute responsibilities and cluster points
        compute_responsibilities_lat(X, c, W, π, γ, h)
        y = classify(γ)
        plot_classes_2d(X, y, 2)

        # 3. M-step: update centroids and mixture coeffs
        N = sum(γ, dims=1)[:]
        c = update_centers(X, γ, N)
        π = N / n

        # 4. triplet loss: update weights W
        for j in 1:100
            anchor, pos, neg, y_p, y_n = select_triplet_vec(X, y, W)
            (y_n == 0) && return c, ones(Int64, n), [w * w' for w in W], history, iters
            state_tree = Flux.setup(opt, W)
            loss, grad = Flux.withgradient(W) do w
                triplet_loss_vec(anchor, pos, neg, y_p, y_n, w)
            end
            Flux.update!(state_tree, W, grad[1])
            push!(history, vcat(W...))
            
        end
        # 5. Convergence check
        weight_diff = sum(sum((W[i] .- last_weights[i]).^2) for i in 1:k)
        centroid_diff = sum((c .- last_centroids).^2)
        (weight_diff < 1e-4 && centroid_diff < 1e-4) && break
        # log_likelihood = sum(log(sum(π[j] * gaussian(X[:, i], c[:, j], W[j] * W[j]') for j in 1:k)) for i in 1:n)
        # diff = abs(log_likelihood - loglike_init)
        # println("Iteration $iter, log likelihood  $log_likelihood, params = $W")
        # (diff < 0.02) && break
        # loglike_init = log_likelihood
        
    end

    weights = [w * w' for w in W]
    return c, y, weights, γ, history, iters
end
