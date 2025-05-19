function split_classes(X, y, anchor, anchor_label)
    """
    Based on an anchor splits the classes into positive samples (nodes of 
    the same class as the anchor) and negative samples (opposite class)

    Params:
    X (Vector{ProductNode}): data (ProductNodes)
    y (Vector{Int64}): labels
    anchor (ProductNode): an anchor
    anchor_label: an anchor's label

    Return:
    (Vector{ProductNode}, Vector{ProductNode}): tuple of positives and negatives
    """
    positives = [X[j] for j in 1:length(X) if y[j] == anchor_label && X[j] != anchor]
    negatives = [X[j] for j in 1:length(X) if y[j] != anchor_label]
    return positives, negatives
end

function htd(pn1, pn2, metric)
    return only(metric(pn1, pn2))
end

function _mahalanobis_vec(x, y, w)
    return w[1]^2 * (x[1] - y[1])^2 + w[2]^2 * (x[2] - y[2])^2
end

function _mahalanobis_mtx(x, y, W)
    diff = x - y
    Σ = W * W'
    return dot(diff, Σ \ diff)
end

function select_triplet_vec(X, y::Vector, W)
    d, n  = size(X)
    perm = Random.randperm(n)
    anchor, pos, neg = perm[1], 0, 0
    anchor_label = y[anchor]
    d_pos = 0.0
    d_neg = Inf64
    for i in perm
        if (y[i] == anchor_label && i != anchor)
            _d = _mahalanobis_vec(X[:, anchor], X[:, i], W[y[i]])
            if (_d > d_pos)
                d_pos = _d
                pos = i
            end
        end
        if (y[i] != anchor_label)
            _d = _mahalanobis_vec(X[:, anchor], X[:, i], W[y[i]])
            if (_d < d_neg)
                d_neg = _d
                neg = i
            end
        end
    end
    (neg == 0) && return zeros(d), zeros(d), zeros(d), 0, 0
    return X[:, anchor], X[:, pos], X[:, neg], y[pos], y[neg]
end

function select_triplet_lat_vec(X::Matrix, c::Matrix, y::Vector, W)
    d, n  = size(X)
    k = size(c, 2)
    perm = Random.randperm(n)
    anchor, pos, neg = rand(1:k, 1, 1)[1], 0, 0
    anchor_label = y[anchor]
    d_pos = 0.0
    d_neg = Inf64
    for i in perm
        if (y[i] == anchor_label && i != anchor)
            _d = _mahalanobis_vec(c[:, anchor], X[:, i], W[y[i]] * W[y[i]]')
            if (_d > d_pos)
                d_pos = _d
                pos = i
            end
        end
        if (y[i] != anchor_label)
            _d = _mahalanobis_vec(c[:, anchor], X[:, i], W[y[i]] * W[y[i]]')
            if (_d < d_neg)
                d_neg = _d
                neg = i
            end
        end
    end
    (neg == 0) && return zeros(d), zeros(d), zeros(d), 0, 0
    return c[:, anchor], X[:, pos], X[:, neg], y[pos], y[neg]
end

function _mahalanobis_pn(pn1, pn2, w)
    x1 = only(pn1.data.x.data)
    y1 = only(pn1.data.y.data)
    x2 = only(pn2.data.x.data)
    y2 = only(pn2.data.y.data)
    return w[1]^2 * (x1 - x2)^2 + w[2]^2 * (y1 - y2)^2
end

function pairwise_distance(X, y, W)
    """
    Computes pairwise distances between ProductNodes

    Params:
    X (Vector{ProductNode}): an array of ProductNodes (dataset)
    wt (WeightStruct): weight transform (identity/softmax)

    Return:
    Matrix{Float64}: matrix with pairwise distances 
    """
    n = size(X, 2)
    (n == 0) && return zeros(Float64, 1, 1)
    distances = zeros((n, n))
    for i in 1:n
        for j in 1:n
            (i != j) && (distances[i, j] = _mahalanobis_mtx(X[:, i], X[:, j], W[y[i]]))
        end
    end
    return distances
end

function pairwise_distance_htd(X, y, metric; wt=identity)
    """
    Computes pairwise distances between ProductNodes

    Params:
    X (Vector{ProductNode}): an array of ProductNodes (dataset)
    wt (WeightStruct): weight transform (identity/softmax)

    Return:
    Matrix{Float64}: matrix with pairwise distances 
    """
    n = length(X)
    (n == 0) && return zeros(Float64, 1, 1)
    distances = zeros((n, n))
    for i in 1:n
        for j in 1:n
            (i != j) && (distances[i, j] = htd(X[i], X[j], metric[y[i]]))
        end
    end
    return distances
end

function select_triplet_lat_htd(X, c, y, W)
    """
    For random k product nodes finds a triplet (the nearest neg 
    and the farthest pos from the dataset)

    Params:
    dists (Matrix{Float64}): matrix with pairwise distances
    X (Vector{ProductNode}): an array of ProductNodes (dataset)
    y (Vector{int}): labels 
    metric (reflectmetric)

    Return:
    (ProductNode, ProductNode, ProductNode): the triplet 
    """
    n = length(y)
    k = length(c)
    perm = Random.randperm(n)
    anchor, pos, neg = rand(1:k, 1, 1)[1], 0, 0
    anchor_label = y[anchor]
    d_pos = 0.0
    d_neg = Inf64
    for i in perm
        if (y[i] == anchor_label && i != 1)
            d = htd(X[anchor], X[i], W[y[i]])
            if (d > d_pos)
                d_pos = d
                pos = i
            end
        end
        if (y[i] != anchor_label)
            d = htd(X[anchor], X[i], W[y[i]])
            if (d < d_neg)
                d_neg = d
                neg = i
            end
        end
    end
    (neg == 0) && return zeros(10), zeros(10), zeros(10), 0, 0
    return c[anchor], X[pos], X[neg], y[pos], y[neg]
end

function select_triplet_htd(::SelectRandom, X, y, w)
    """
    Every training iteration selects the triplet randomly. 

    Params:
    dists (Matrix{Float64}): matrix with pairwise distances
    X (Vector{ProductNode}): an array of ProductNodes (dataset)
    y (Vector{int}): labels 
    metric (reflectmetric)

    Return:
    (ProductNode, ProductNode, ProductNode): the triplet 
    """ 
    n = length(y)
    perm = Random.randperm(n)
    anchor, pos, neg = perm[1], 0, 0
    anchor_label = y[anchor]
    for i in perm
        (y[i] != anchor_label || i != anchor) && continue
        pos = i
        for j in perm
            d_pos = htd(X[anchor], X[pos], w)
            d_neg = htd(X[anchor], X[j], w)
            if (y[j] != anchor_label && d_pos < d_neg)
                neg = j
                break
            end
        end
        (neg != 0) && break
    end
    return X[anchor], X[pos], X[neg], y[pos], y[neg]
end

function select_triplet_htd(::SelectHard, X, y, w)
    """
    For random k product nodes finds a triplet (the nearest neg 
    and the farthest pos from the dataset)

    Params:
    dists (Matrix{Float64}): matrix with pairwise distances
    X (Vector{ProductNode}): an array of ProductNodes (dataset)
    y (Vector{int}): labels 
    metric (reflectmetric)

    Return:
    (ProductNode, ProductNode, ProductNode): the triplet 
    """
    n = length(y)
    perm = Random.randperm(n)
    anchor, pos, neg = perm[1], 0, 0
    anchor_label = y[anchor]
    d_pos = 0.0
    d_neg = Inf64
    for i in perm
        if (y[i] == anchor_label && i != 1)
            d = htd(X[1], X[i], w)
            if (d > d_pos)
                d_pos = d
                pos = i
            end
        end
        if (y[i] != anchor_label)
            d = htd(X[1], X[i], w)
            if (d < d_neg)
                d_neg = d
                neg = i
            end
        end
    end
    return X[anchor], X[pos], X[neg], y[pos], y[neg]
end

function triplet_loss_htd(a, p, n, y_p, y_n, W; α = 0.3, λₗₐₛₛₒ = 0.01, weight_transform=identity)
    """
    Computes the triplet loss function with Lasso (L1) regularization

    Params:
    a (ProductNode): an anchor
    p (ProductNode): positive sample
    n (ProductNode): negative sample
    metric (reflectmetric): metric used for distance calculation
    α (Float64): a bias used in triplet loss function
    λₗₐₛₛₒ (Float64): Lasso parameter
    weight_transform (WeightStruct): weight transform (identity/softmax)

    Return:
    (Float64): the loss
    """
    d_pos = htd(a, p, W[y_p])
    d_neg = htd(a, n, W[y_n])
    f(x) = weight_transform(x)
    p = [softplus.(Flux.destructure(w)[1] for w in W)]
    _params = reduce(vcat, reduce(vcat, p))
    reg = sum(x->sum(abs.(f(x))), _params)
    return max(d_pos - d_neg + α, 0) + λₗₐₛₛₒ * reg
end

function triplet_loss_vec(a, p, n, y_p, y_n, W; α = 0.5, λₗₐₛₛₒ = 0.01)
    d_pos = _mahalanobis_mtx(a, p, W[y_p] * W[y_p]')
    d_neg = _mahalanobis_mtx(a, n, W[y_n] * W[y_n]')
    reg = sum(sum(abs, i) for i in W)
    return max(d_pos - d_neg + α, 0) + λₗₐₛₛₒ * reg
end
