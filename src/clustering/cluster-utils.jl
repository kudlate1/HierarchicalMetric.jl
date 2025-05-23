function set_centroids_lac(X::Matrix, y::Vector, k::Int)
    """
    Computes new centroids based on clustering S.

    Params:
    X (Matrix): points
    y (Vector): labels
    k (Int):    number of clusters (centroids respectively)
    d (Int):    number of dimensions

    Return:
    (Matrix): updated centroids
    """
    d = size(X, 1)
    centroids = zeros(d, k)
    for i in 1:k
        assigned = [x for (x_i, x) in enumerate(eachcol(X)) if vec(y)[x_i] == i]
        numerator = (!isempty(assigned)) ? sum(assigned) : zero(X[:, 1])
        denominator = count(==(i), vec(y))
        centroids[:, i] = numerator ./ denominator
    end
    return centroids
end

function set_weights_lac(X::Matrix, c::Matrix, y::Vector, h::Float64)
    """
    Computes weights from variances calculated also in this function.

    Params:
    c (Matrix): current centroids
    S (Dict): current subsets containing vectors of each cluster
    h (Float64)

    Return:
    (Matrix): new weights
    """
    d, k = size(c)
    variances = zeros(d, k)
    for i in 1:k
        s_i = X[:, vec(y) .== i]
        variances[:, i] = mean(((c[:, i] .- s_i).^2), dims=2)
    end
    exp_x = exp.(-variances ./ h)
    weights = exp_x ./ sum(exp_x, dims=1)
    return weights
end

function update_centers_lat(X, idxs, y, W)
    distances = pairwise_distance(X, y, W)
    result = kmedoids!(distances, idxs)
    return result.medoids
end

function update_centers_lat_htd(X, idxs, y, W)
    distances = pairwise_distance_htd(X, y, W)
    result = kmedoids!(distances, idxs)
    return result.medoids
end

function Lw(cₗ, xᵢ, wₗ)
    return √sum(wₗ.^2 .* (cₗ - xᵢ).^2) 
end

function indicator(xi_label::Int, j::Int)
    """
    Params:
    xi_label: current cluster of xᵢ
    j: subset (cluster)

    Return:
    true if xi_label == j else false
    """
    return (xi_label == j) ? 1 : 0
end

function gaussian(xₙ, μₖ, Σₖ)
    """
    Computes the probability density for a particular xₙ, μₖ and Σₖ.

    xₙ: n-th observation point
    μₖ: the mean vector of cluster k
    Σₖ: the covariance matrix for cluster k

    Return:
    xₙ, μₖ, Σₖ: initialized parameters
    """
    d = size(xₙ, 1)
    frac1 = 1 / (2π)^(d/2)
    frac2 = 1 / sqrt(det(Σₖ))
    _exp = -0.5 * (xₙ - μₖ)' * inv(Σₖ) * (xₙ - μₖ)
    return frac1 * frac2 * exp(_exp)
end

function compute_responsibilities_gmm(X, μ, Σ, π, γ)
    n = size(X, 2)
    k = size(μ, 2)
    for i in 1:n
        total_prob = 0.0
        for j in 1:k
            γ[i, j] = π[j] * gaussian(X[:, i], μ[:, j], Σ[j])
            total_prob += γ[i, j]
        end
        γ[i, :] ./= total_prob
    end
    return γ
end

function compute_responsibilities_htd(X, c, W, π, γ, h)
    n = length(X)
    k = length(c)
    for i in 1:n
        total_prob = 0.0
        for j in 1:k
            γ[i, j] = π[j] * exp(-htd(X[i], c[j], W[j]) / h)
            total_prob += γ[i, j]
        end
        γ[i, :] ./= total_prob
    end
end

function update_centers(X, γ, N)
    k = length(N)
    d = size(X, 1)
    c = zeros(d, k)
    for j in 1:k
        c[:, j] = sum(X .* γ[:, j]', dims=2) / N[j]
    end
    return c
end

function update_covariances(X, μ, γ, N)
    d, n = size(X)
    k = size(μ, 2)
    Σ = [Matrix(1.0I, d, d) for _ in 1:k]
    for j in 1:k
        Σ[j] = zeros(d, d)
        for i in 1:n
            diff = X[:, i] - μ[:, j]
            Σ[j] += γ[i, j] * (diff * diff')
        end
        Σ[j] /= N[j]
    end
    return Σ
end

function classify(γ)
    """
    Cluster points based on the soft assignments (posterior probabilities γ).
    The point is assigned to a cluster with the higher posterior probability. 
    """
    return ifelse.(γ[:, 1] .<= 0.5, 2, 1)
end

function compute_responsibilities_lat(X, c, W, π, γ, h)
    d, n = size(X)
    k = size(c, 2)
    for i in 1:n
        total_prob = 0.0
        for j in 1:k
            γ[i, j] = π[j] * exp(-_mahalanobis_mtx(X[:, i], c[:, j], W[j] * W[j]') / h)
            total_prob += γ[i, j]
        end
        γ[i, :] ./= total_prob
    end
end

function init_centers(::Kmeanspp, X::Matrix, k::Int)
    """
    Performs k-means++ initialization for clustering.
    """
    n, m = size(X)
    centroids = zeros(n, k)
    # 1. select the first centroid randomly
    c1 = rand(1:m)
    centroids[:, 1] = X[:, c1]
    # 2. initialize distance array
    pl_distribution = fill(Inf, m)
    for i in 2:k
        # 3. compute squared Euclidean distances from each point to closest centroid
        for j in 1:m
            pl_distribution[j] = min(pl_distribution[j], sum((X[:, j] .- centroids[:, i-1]) .^ 2))
        end
        # 4. choose next centroid with probability proportional to distance squared
        idx = rand(Categorical(pl_distribution ./ sum(pl_distribution)))
        centroids[:, i] = X[:, idx]
    end
    return centroids
end

function init_centers(::FarthestPoint, X::Matrix, k::Int)
    """
    The first point is chosen randomly from the dataset, next 
    center is the farthest from the existing ones.
    """
    d, n = size(X)
    centroids = zeros(d, k)
    selected = falses(n)
    # 1. pick one point at random
    first_idx = rand(1:n)
    centroids[:, 1] = X[:, first_idx]
    selected[first_idx] = true
    # 2. select k-1 farthest points
    for i in 2:k
        max_dist = -Inf
        next_idx = 0
        for j in 1:n
            selected[j] && continue
            dists = [sum((X[:, j] .- centroids[:, c]).^2) for c in 1:(i-1)]
            min_dist = minimum(dists)
            if min_dist > max_dist
                max_dist = min_dist
                next_idx = j
            end
        end
        centroids[:, i] = X[:, next_idx]
        selected[next_idx] = true
    end
    return centroids
end

function init_centers(::RandomPoint, X::Matrix, k::Int)
    n = size(X, 2)
    return X[:, rand(1:n, k)]
end

function weight_transform(X::Matrix, y, weights::Matrix)
    X = [x .* weights[:, y[x_i]] for (x_i, x) in enumerate(eachcol(X))]
    return hcat(X...)
end

function metrics(X, true_labels, clusters)
    vm = vmeasure(true_labels, clusters)
    var = varinfo(true_labels, clusters)
    rand = randindex(true_labels, clusters)
    return vm, var, rand
end

squared_distance(x, y) = sum((x .- y).^2)
_precision(_found, _true, d) = mean([squared_distance(_found[:, i], _true[:, i]) for i in 1:d])