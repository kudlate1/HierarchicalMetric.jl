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

function distance(pn1, pn2, metric)
    return only(metric(pn1, pn2))
end

function pairwise_distance(X; wt=identity)

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
    metric = reflectmetric(X[1], weight_transform=wt)

    for i in 1:n
        for j in 1:n
            (i != j) && (distances[i, j] = distance(X[i], X[j], metric))
        end
    end
       
    return distances
end

function _mahalanobis_pn(pn1, pn2, w)
 
    x1 = only(pn1.data.x.data)
    y1 = only(pn1.data.y.data)
    x2 = only(pn2.data.x.data)
    y2 = only(pn2.data.y.data)
    return w[1]^2 * (x1 - x2)^2 + w[2]^2 * (y1 - y2)^2
end

function select_triplet(::SelectRandom, X, y, w)

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
        if (y[i] == anchor_label && i != anchor)
            pos = i
            for j in perm
                d_pos = distance(X[anchor], X[pos], w)
                d_neg = distance(X[anchor], X[j], w)
                if (y[j] != anchor_label && d_pos < d_neg)
                    neg = j
                    break
                end
            end
            (neg != 0) && break
        end
    end

    return X[anchor], X[pos], X[neg]
end

function select_triplet(::SelectHard, X, y, w)

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
    anchor, pos, neg = 1, 0, 0
    anchor_label = y[1]

    d_pos = 0.0
    d_neg = Inf64

    for i in perm

        if (y[i] == anchor_label && i != 1)
            d = distance(X[1], X[i], w)
            if (d > d_pos)
                d_pos = d
                pos = i
            end
        end
        if (y[i] != anchor_label)
            d = distance(X[1], X[i], w)
            if (d < d_neg)
                d_neg = d
                neg = i
            end
        end
    end

    return X[anchor], X[pos], X[neg]
end

function triplet_loss(a, p, n, metric; α = 0.3, λₗₐₛₛₒ = 0.1, weight_transform=identity)

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

    d_pos = distance(a, p, metric)
    d_neg = distance(a, n, metric)
    
    f(x) = weight_transform(x)
    w = Flux.params(metric) 
    # mutagenesis weigts .... 
    # w = Params([Float32[1.0, 1.0, 1.0, 1.0], Float32[1.0, 1.0, 1.0, 1.0], Float32[1.0, 1.0, 1.0, 1.0, 1.0]])   
    # w = [x₁, x₂, x₃] = [[x₁₁, x₁₂, x₁₃, x₁₄], ..., [x₃₁, x₃₂, x₃₃, x₃₄, x₃₅]]
    # L1 regularization 
    # reg = √(∑ᵢ(∑ⱼ xᵢⱼ²)) ---->   √(x₁₁² + x₁₂² + ... +  x₃₄² + x₃₅²)
    # f(x) is weight transform of each element of weights

    reg = sum(x->sum(abs.(f(x))), w)

    return max(d_pos - d_neg + α, 0)  #+ λₗₐₛₛₒ * reg
end
