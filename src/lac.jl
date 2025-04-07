function set_centroids(X::Matrix, labels, k::Int, d::Int)
    """
    Computes new centroids based on clustering S.

    Params:
    X (Matrix): points
    S (Dict): current subsets containing vectors of each cluster
    k (Int): number of clusters (centroids respectively)
    d (Int): number of dimensions

    Return:
    (Matrix): new centroids
    """

    centroids = zeros(d, k)

    for i in 1:k
        numerator = sum(x for (x_i, x) in enumerate(eachcol(X)) if vec(labels)[x_i] == i)
        denominator = count(==(i), vec(labels))
        centroids[:, i] = numerator ./ denominator
    end

    return centroids
end

function set_weights(X, centroids::Matrix, labels::Matrix, h::Float64, k::Int, d::Int)
    """
    Computes weights from variances calculated also in this function.

    Params:
    centroids (Matrix): current centroids
    S (Dict): current subsets containing vectors of each cluster
    h (Float64)

    Return:
    (Matrix): new weights
    """

    variances = zeros(d, k)

    for i in 1:k
        s_i = X[:, vec(labels) .== i]
        variances[:, i] = mean(((centroids[:, i] .- s_i).^2), dims=2)
    end
    
    exp_x = exp.(-variances ./ h)
    weights = exp_x ./ sum(exp_x, dims=1)

    return weights
end

function Lw(cₗ, xᵢ, wₗ)

    return √sum(wₗ.^2 .* (cₗ - xᵢ).^2) 
end

function indicator(xi_label::Int, j::Int)
    """
    Params:
    xi_label: current xᵢ cluster
    j: subset (cluster)

    Return:
    true if xi_label == j else false
    """

    return (xi_label == j) ? 1 : 0
end

function kmeanspp(X, k::Int)
    """
    Performs k-means++ initialization for k-means/LAC clustering.

    Params:
    X: feature vectors
    k: required number of clusters

    Return:
    centroids:  proposed centroids for k-means initialization
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

function LAC(X::Matrix, k::Int; max_iter::Int=50, h::Float64=10.0)
    """
    Performs LAC algorithm.

    Params:
    X (Matrix):         points
    c (Matrix):         initial centroids
    k (Int):            number of clusters (centroids respectively)
    d (Int):            number of dimensions
    max_iter (Int):     max number of iterations
    h (Float64)

    Return:
    (Matrix): trained centroids
    """

    d, n = size(X) # dims, points

    # 1., 2. init centroids, weights and labels
    centroids = kmeanspp(X, k)
    #centroids = X[:, rand(1:n, k)] # rnd points
    weights = 1/d * ones(d, k)
    labels = -1 * ones(Int, 1, n)

    for iter in 1:max_iter

        last_weights = weights
        last_centroids = centroids

        # 3. sort points into the closest clusters
        for (l, x_l) in enumerate(eachcol(X))
            distances = [Lw(centroids[:, i], x_l, weights[:, i]) for i in 1:k]
            j = argmin(distances)
            labels[l] = j
        end

        # 4. compute new weights
        weights = set_weights(X, centroids, labels, h, k, d)
        
        # 5. resort points into clusters Sⱼ
        for (l, x_l) in enumerate(eachcol(X))
            distances = [Lw(centroids[:, i], x_l, weights[:, i]) for i in 1:k]
            j = argmin(distances)
            labels[l] = j
        end

        # 6. recompute centroids
        centroids = set_centroids(X, labels, k, d)

        # print + convergence check
        weight_diff = sum((weights .- last_weights).^2)
        centroid_diff = sum((centroids .- last_centroids).^2)

        println("Iteration $iter: weight change = $weight_diff, centroid change = $centroid_diff")
        (weight_diff <= 1e-4 && centroid_diff <= 1e-4) && break
    end

    return centroids, weights, labels
end
