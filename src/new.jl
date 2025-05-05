function _mahalanobis(x, y, w)

    return w[1] * (x[1] - y[1])^2 + w[2] * (x[2] - y[2])^2
end

function select_triplet_vec(X, y, W)
    
    _, n  = size(X)

    perm = Random.randperm(n)
    anchor, pos, neg = perm[1], 0, 0
    anchor_label = y[anchor]

    for i in perm
        (y[i] != anchor_label || i == anchor) && continue
        pos = i
        for j in perm
            (y[j] == anchor_label) && continue
            d_pos = _mahalanobis(X[:, anchor], X[:, i], W[:, y[i]])
            d_neg = _mahalanobis(X[:, anchor], X[:, j], W[:, y[j]])
            if (d_pos < d_neg)
                neg = j
                break
            end
        end
        (neg != 0) && break
    end

    return X[:, anchor], X[:, pos], X[:, neg], y[pos], y[neg]
end

function triplet_loss_vec(a, p, n, y_p, y_n, W; α = 0.1, λₗₐₛₛₒ = 0.1)

    d_pos = _mahalanobis(a, p, W[:, y_p])
    d_neg = _mahalanobis(a, n, W[:, y_n])
    reg = sum(abs.(hcat(W...)))
    return max(d_pos - d_neg + α, 0) + λₗₐₛₛₒ * reg
end

function update_centroids(X, c, γ, N)

    k = size(c, 2)

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

function train_new(X, k; h=5.0, λ=0.1, max_iter=50)

    d, n = size(X)

    # 1. init
    c = kmeanspp(X, k)
    γ = zeros(n, k)
    W = ones(d, k)  # each cluster has a params w₁ and w₂ (for each dim)
    π = fill(1/k, k)
    y = zeros(Int64, n)

    opt = Adam(λ)
    history = []
    old_loss = Inf

    for iter in 1:max_iter

        # 2. E-step: compute responsibilities and cluster points
        compute_responsibilities_new(X, c, W, π, γ, h)
        y = cluster_points(γ, y)

        # 3. M-step: update centroids and mixture coeffs
        N = sum(γ, dims=1)[:]
        c = update_centroids(X, c, γ, N)
        π = N / n

        # 4. triplet loss: update weights W
        anchor, pos, neg, y_p, y_n = select_triplet_vec(X, y, W)
        state_tree = Flux.setup(opt, W)
        loss, grad = Flux.withgradient(W) do w
            triplet_loss_vec(anchor, pos, neg, y_p, y_n, w)
        end
        Flux.update!(state_tree, W, grad[1])
        # parameters are problematic

        # 5. Convergence check
        push!(history, hcat(W...))
        println("Iteration $iter, loss $loss, params = $W")
        (abs(old_loss - loss) < 1e-5) && break
        old_loss = loss
    end

    return c, y, weights, history
end

### TODO: restructure the project ###

"""
X, y = generate_separable_2d(GaussianData(), 50, 50, 
          [-1.0, 0.0], [1.0, 0.0], [0.1 0.0; 0.0 10.0], [0.1 0.0; 0.0 10.0])

ps, y_new, h = train_new(X, 2);
plot_classes_2d(X, y_new, 2)
"""