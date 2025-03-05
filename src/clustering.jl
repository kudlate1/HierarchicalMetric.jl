function setCentroids(X::Matrix, S::Dict, k::Int, d::Int)

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
        numerator = (sum(x .* indicator(x, S["S$i"]) for x in eachcol(X)))
        denominator = size(S["S$i"], 1) # much faster
        centroids[:, i] = numerator ./ denominator
    end

    return centroids
end

function initSubsets(k::Int)

    """
    Initializes partition S and its 'k' subsets Sⱼ. S is represented as dictionary with keys Sⱼ in 
    pair with an empty array S = Dict("S1" => [], ..., "Sk" => []).

    Later the vectors are pushed into corresponding subsets Sⱼ ->  S = Dict("Sj" => Vector{Vector{}}).
    After sorting all the 'n' points, the Vector{Vector{}} structure is transformed into Matrix in
    order to maintain the consistency of used data types.

    Params:
    k (Int): number of clusters

    Return:
    (Dict): empty Initialized dictionary S = Dict("S1" => [], ..., "Sk" => [])
    """

    S = Dict()
    for j in 1:k
        push!(S, "S$j" => [])
    end

    return S
end

function setWeights(centroids::Matrix, S::Dict, h::Float64, k::Int, d::Int)

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
        variances[:, i] = sum([((centroids[:, i] - x).^2) ./ length(S["S$i"]) for x in eachcol(S["S$i"])])
    end
    
    expX = exp.(-variances ./ h)
    weights = expX ./ sum(expX, dims=1)

    return weights
end

function Lw(cₗ, xᵢ, wₗ)
    
    return √sum(wₗ .* (cₗ - xᵢ).^2)
end

function indicator(x, Sⱼ::Matrix)

    """
    Params:
    x (Matrix): d×1 matrix (column of X)
    Sⱼ (Matrix): 
    """

    x in eachcol(Sⱼ) # returns true if x is in Sⱼ else false
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
      
        normXi = vec(sum(X .^ 2, dims=1))
        normXj = sum(centroids[:, 1:i-1] .^ 2, dims=1)
        dl_2 = minimum(normXj .+ normXi' .- 2 .* (X' * centroids[:, 1:i-1]), dims=2)[:]
        pl_distribution = min.(pl_distribution, dl_2)
        idx = rand(Categorical(pl_distribution ./ sum(pl_distribution)))
        centroids[:, i] = X[:, idx]
    end

    return centroids

end

function LAC(X::Matrix, c::Matrix, k::Int, d::Int; max_iter::Int=20, h::Float64=0.5)

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

    # 1., 2. init centroids and weights
    centroids = kmeanspp(X, k)
    weights = 1/d * ones(d, k)

    for iter in 1:max_iter

        lastWeights = weights

        # 3. sort points into the closest clusters
        S = initSubsets(k)
        for x in eachcol(X)
            j = argmin(Lw(centroids[:, i], x, weights[:, i]) for i in 1:k)
            push!(S["S$j"], x)
        end
        S = Dict(["S$i" => hcat(S["S$i"]...) for i in 1:k])

        # 4. compute new weights
        weights = setWeights(centroids, S, h, k, d)
        
        # 5. resort points into clusters Sⱼ
        S = initSubsets(k)
        for x in eachcol(X)
            j = argmin(Lw(centroids[:, i], x, weights[:, i]) for i in 1:k)
            push!(S["S$j"], x)
        end
        S = Dict(["S$i" => hcat(S["S$i"]...) for i in 1:k])

        # 6. recompute centroids
        centroids = setCentroids(X, S, k, d)

        # print + convergence check
        condition = sum((weights .- lastWeights).^2) # original condition did not make sense to me, I changed it to L2
        println("iteration $iter: condition $condition: centroids $centroids, weights $weights")
        (condition <= 1e-4) && break
    end

    return centroids, weights
end

"""
1) dopsat LAC do overleafu
2) přepsat experimenty, datasety
3) experiment s clusteringem (nezná třídy), vyhodnotit kvalitu
4) metriky (ari, vmeasure, ...) - Clustering.jl

- do přepsání experimentů přidat dataset, kde každý cluster jinou metriku + otestovat 
  clustering s globální a lokální metrikou
  
"""