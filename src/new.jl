function _mahalanobis(x, y, w)

    return w[1]^2 * (x[1] - y[1])^2 + w[2]^2 * (x[2] - y[2])^2
end

function select_triplet_vec(X, y, W)
    
    _, n  = size(X)

    perm = Random.randperm(n)
    anchor, pos, neg = perm[1], 0, 0
    anchor_label = y[anchor]

    d_pos = 0.0
    d_neg = Inf64

    for i in perm

        if (y[i] == anchor_label && i != anchor)
            d = _mahalanobis(X[:, anchor], X[:, i], W[:, y[i]])
            if (d > d_pos)
                d_pos = d
                pos = i
            end
        end
        if (y[i] != anchor_label)
            d = _mahalanobis(X[:, anchor], X[:, i], W[:, y[i]])
            if (d < d_neg)
                d_neg = d
                neg = i
            end
        end
    end

    return X[:, anchor], X[:, pos], X[:, neg], y[pos], y[neg]
end

function triplet_loss_vec(a, p, n, y_p, y_n, W; α = 0.1, λₗₐₛₛₒ = 0.3)

    d_pos = _mahalanobis(a, p, W[:, y_p])
    d_neg = _mahalanobis(a, n, W[:, y_n])
    reg = sum(abs, W)
    return max(d_pos - d_neg + α, 0) + λₗₐₛₛₒ * reg
end

function update_centroids(X, γ, N)

    k = length(N)
    d = size(X, 1)
    c = zeros(d, k)

    for j in 1:k
        c[:, j] = sum(X .* γ[:, j]', dims=2) / N[j]
    end
    return c
end

function cluster_points(γ, y)

    return y .= ifelse.(γ[:, 1] .<= 0.5, 2, 1)
end

function compute_responsibilities_new(X, c, W, π, γ, h)

    n = size(X, 2)
    k = size(c, 2)

    for i in 1:n
        total_prob = 0.0
        for j in 1:k
            γ[i, j] = π[j] * exp(-_mahalanobis(X[:, i], c[:, j], W[:, j]) / h)
            total_prob += γ[i, j]
        end
        γ[i, :] ./= total_prob
    end
end

function train_new(X, k; h=1.0, λ=0.001, max_iter=50)

    """
    Clustering poměrně dobrý
    Jak poznám, kdy skončit? -> rozdíl old a new sillhouttes nebo tak něco?...
    """

    d, n = size(X)

    # 1. init
    #c = kmeanspp(X, k)
    c = X[:, rand(1:n, k)]
    γ = zeros(n, k)
    W = ones(d, k)  # each cluster has a params w₁ and w₂ (for each dim)
    π = fill(1/k, k)
    y = zeros(Int64, n)

    opt = Adam(λ)
    history = []

    for iter in 1:max_iter

        last_weights = W
        last_centroids = c

        # 2. E-step: compute responsibilities and cluster points
        compute_responsibilities_new(X, c, W, π, γ, h)
        y = cluster_points(γ, y)
        plot_classes_2d(X, y, 2)

        # 3. M-step: update centroids and mixture coeffs
        N = sum(γ, dims=1)[:]
        c = update_centroids(X, γ, N)
        π = N / n

        # 4. triplet loss: update weights W
        for j in 1:5
            anchor, pos, neg, y_p, y_n = select_triplet_vec(X, y, W)
            state_tree = Flux.setup(opt, W)
            loss, grad = Flux.withgradient(W) do w
                triplet_loss_vec(anchor, pos, neg, y_p, y_n, w)
            end
            Flux.update!(state_tree, W, grad[1])

            # 5. Convergence check
            push!(history, vcat(W...))
            println("Iteration $iter.$j, loss $loss, anchor $anchor, params = $W")
        end

        weight_diff = sum((W .- last_weights).^2)
        centroid_diff = sum((c .- last_centroids).^2)
        (weight_diff <= 1e-4 && centroid_diff <= 1e-4) && break
    end

    return c, y, W, history
end

### TODO: restructure the project ###

"""
X, y = generate_separable_2d(GaussianData(), 50, 50, 
          [-3.0, 0.0], [3.0, 0.0], [1.0 0.0; 0.0 1.0], [1.0 0.0; 0.0 1.0])

centroids, y_new, weights, h = train_new(X, 2);
plot_classes_2d(X, y_new, 2)
plot_process(weights, h)

X_transformed = weight_transform(X, y_new, weights)
plot_classes_2d(X_transformed, y, 2)
"""