function setCentroids(X::Matrix, S::Dict, k::Int, d::Int)

    centroids = zeros(d, k)

    for i in 1:k
        centroids[:, i] = (sum(x .* indicator(x, S["S$i"]) for x in eachcol(X))) ./ sum(indicator(x, S["S$i"]) for x in eachcol(X))
    end

    return centroids
end

function initSubsets(k::Int)

    S = Dict()
    for i in 1:k
        push!(S, "S$i" => [])
    end

    return S
end

function setWeights(centroids::Matrix, S::Dict)

    weights = zeros(d, k)
    variances = zeros(d, k)

    for i in 1:k
        variances[:, i] = sum([((centroids[:, i] - x).^2) ./ length(S["S$i"]) for x in eachcol(S["S$i"])])
        for l in 1:d
            weights[l, i] = exp(-variances[l, i] / h) / sum([exp(-variances[idx, i] / h) for idx in 1:d])
        end
    end

    return weights
end

function Lw(cₗ, xᵢ, wₗ)
    
    return √sum(wₗ .* (cₗ - xᵢ).^2)
end

function indicator(x, Sⱼ)

    for i in eachcol(Sⱼ)
        (x == i) && return true
    end

    return false
end

# když se inicializujou centroidy tak zle, že nějaké Sⱼ prázdné -> problém
function LAC(X, c, k, d; max_iter=20, h=5.0)

    """
    Performs LAC algorithm.

    Params:
    X (Matrix{Float64}): points
    c (Matrix{Float64}): initial centroids
    k (Int64): number of clusters (centroids respectively)
    d (Int64): number of dimensions
    max_iter (Int64): max number of iterations
    h (Float64)

    Return:
    (Matrix{Float64}): trained centroids

    """

    # init centroids, weightsa and variances
    centroids = c
    weights = 1/d * ones(d, k)

    for iter in 1:max_iter

        # 3. sort points into the closest clusters
        S = initSubsets(k)
        for x in eachcol(X)
            j = argmin(Lw(centroids[:, i], x, weights[:, i]) for i in 1:k)
            push!(S["S$j"], x)
        end
        S = Dict(["S$i" => hcat(S["S$i"]...) for i in 1:k])

        # 4. recompute weights
        weights = setWeights(centroids, S)

        # 5. resort points into Sⱼ
        S = initSubsets(k)
        for x in eachcol(X)
            j = argmin(Lw(centroids[:, i], x, weights[:, i]) for i in 1:k)
            push!(S["S$j"], x)
        end
        S = Dict(["S$i" => hcat(S["S$i"]...) for i in 1:k])

        # 6. recompute centroids
        centroids = setCentroids(X, S, k, d)

        println("iteration $iter: centroids $centroids, weights $weights")
    end

    return centroids
end

# -----------------------------------------------------------------------------------------------
# later in scripts

X = [ 2.5  1.8 -2.1  3.9 -2.5  2.1 ;
    3.2  2.9 -3.1 -2.0 -2.8 -3.3 ]

centroids = [ -2.0  2.0 3.1 ;
-2.0  1.2 -1.1 ]

data = hcat(X, centroids)
labels = [1 1 1 1 1 1 0 0 0]

function plotPoints(data, labels)

    x_coords = vec([i[1] for i in eachcol(data)])
    y_coords = vec([i[2] for i in eachcol(data)])

    colors = [label == 1 ? RGB(0.2, 0.2, 0.8) : RGB(0.4, 0.1, 0.2) for label in vec(labels)]

    scatter(
        x_coords,
        y_coords,
        color = colors,
        marker = (10, :circle),
        xlabel = "x",
        ylabel = "y",
        background_color = RGB(0.2, 0.2, 0.2),
        legend = false
    )
end

plotPoints(data, labels)

newCentroids = LAC(X, centroids, 3, 2)

plotPoints(hcat(X, newCentroids), labels)

