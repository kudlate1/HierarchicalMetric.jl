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
        numerator = (sum(x .* indicator(labels[x_i], i) for (x_i, x) in enumerate(eachcol(X))))
        denominator = sum((i == j) && 1 for j in labels)
        centroids[:, i] = numerator ./ denominator
    end

    return centroids
end

function set_weights(centroids::Matrix, labels::Matrix, h::Float64, k::Int, d::Int)
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
        lenSi = sum((i == j) && 1 for j in labels)
        variances[:, i] = sum([((centroids[:, i] - x).^2) ./ lenSi for x in eachcol(X[])])
    end
    
    exp_x = exp.(-variances ./ h)
    weights = exp_x ./ sum(exp_x, dims=1)

    return weights
end

function Lw(cₗ, xᵢ, wₗ)

    # '√' can be problematic in case of non-softplus'ed values
    return √sum(wₗ .* (cₗ - xᵢ).^2) 
end

function indicator(xi_label::Int, j::Int)
    """
    Params:
    xi_label: current xᵢ cluster
    j: subset (cluster)

    Return:
    true if xilabel == j else false
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
    c1 = rand(1:m)
    centroids[:, 1] = X[:, c1]
    pl_distribution = fill(Inf, m)

    for i in 2:k
      
        norm_xi = vec(sum(X .^ 2, dims=1))
        norm_xj = sum(centroids[:, 1:i-1] .^ 2, dims=1)
        dl_2 = minimum(norm_xj .+ norm_xi' .- 2 .* (X' * centroids[:, 1:i-1]), dims=2)[:]
        pl_distribution = min.(pl_distribution, dl_2)
        idx = rand(Categorical(pl_distribution ./ sum(pl_distribution)))
        centroids[:, i] = X[:, idx]
    end

    return centroids

end

function LAC(X::Matrix, k::Int; max_iter::Int=20, h::Float64=0.5)
    """
    Performs LAC algorithm.

    Params:
    X (Matrix): points
    c (Matrix): initial centroids
    k (Int): number of clusters (centroids respectively)
    d (Int): number of dimensions
    max_iter (Int): max number of iterations
    h (Float64)

    Return:
    (Matrix): trained centroids
    """

    d, _ = size(X) # dims

    # 1., 2. init centroids, weights and labels
    centroids = kmeanspp(X, k)
    weights = 1/d * ones(d, k)
    labels = -1 * ones(1, k)

    for iter in 1:max_iter

        last_weights = weights

        # 3. sort points into the closest clusters
        for (l, x_l) in enumerate(eachcol(X))
            j = argmin(Lw(centroids[:, i], x_l, weights[:, i]) for i in 1:k)
            labels[l] = j
        end

        # 4. compute new weights
        weights = set_weights(centroids, labels, h, k, d)
        
        # 5. resort points into clusters Sⱼ
        for (l, x_l) in enumerate(eachcol(X))
            j = argmin(Lw(centroids[:, i], x_l, weights[:, i]) for i in 1:k)
            labels[l] = j
        end

        # 6. recompute centroids
        centroids = set_centroids(X, labels, k, d)

        # print + convergence check
        condition = sum((weights .- last_weights).^2) # original condition did not make sense to me, I changed it to L2
        println("iteration $iter: condition $condition: centroids $centroids, weights $weights")
        (condition <= 1e-4) && break
    end

    return centroids, weights, labels
end

"""
[ ] 1) dopsat LAC do overleafu
[ ] 2) přepsat experimenty, datasety
[ ] 3) experiment s clusteringem (nezná třídy), vyhodnotit kvalitu
[ ] 4) metriky (ari, vmeasure, ...) - Clustering.jl
[ ] 5) do přepsání experimentů přidat dataset, kde každý cluster jinou metriku + otestovat 
    clustering s globální a lokální metrikou
"""