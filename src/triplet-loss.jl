function splitClasses(X, y, anchor, anchor_label)

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

function pairwiseDistance(X; wt=identity)

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

function myMahalanobis(pn1, pn2, w)
 
    x1 = only(pn1.data.x.data)
    y1 = only(pn1.data.y.data)
    x2 = only(pn2.data.x.data)
    y2 = only(pn2.data.y.data)
    
    return w[1]^2 * (x1 - x2)^2 + w[2]^2 * (y1 - y2)^2
end

function selectTriplet(::SelectRandom, dists, X, y, metric)

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
    
    perm = Random.randperm(length(X))

    for i in perm

        anchor = X[i]
        anchor_label = y[i]
        positives, negatives = splitClasses(X, y, anchor, anchor_label)
        
        if (length(positives) != 0)
            pos = rand(positives)
            neg = (length(negatives) != 0) && rand(negatives)
            return anchor, pos, neg
        end
    end
end

function selectTriplet(::SelectHard, dists, X, y, metric)

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

    k = 10
    n = length(X)
    perm = Random.randperm(n)
    triplets = []

    for i in perm

        (k <= 0) && break 

        anchor = X[i]
        anchor_label = y[i]
        positives, negatives = splitClasses(X, y, anchor, anchor_label)

        (length(positives) == 0) && continue

        positive = false
        negative = false

        d_pos = 0.0
        d_neg = Inf64

        for pos in 1:n
            if (X[pos] in positives && dists[i, pos] > d_pos)
                d_pos = dists[i, pos]
                positive = X[pos]
            end
        end

        for neg in 1:n
            if (X[neg] in negatives && dists[i, neg] < d_neg)
                d_neg = dists[i, neg]
                negative = X[neg]
            end
        end

        k = k-1

        push!(triplets, (anchor, positive, negative))
    end

    return triplets[rand(1:length(triplets))]
end

function tripletLoss(anchor, positive, negative, metric; α = 0.01, λₗₐₛₛₒ = 0.01, weight_transform=identity)

    """
    Computes the triplet loss function with Lasso (L1) regularization

    Params:
    anchor (ProductNode): an anchor
    positive (Vector{ProductNode}): positive samples
    negative (Vector{ProductNode}): negative samples
    metric (reflectmetric): metric used for distance calculation
    α (Float64): a bias used in triplet loss function
    λₗₐₛₛₒ (Float64): Lasso parameter
    weight_transform (WeightStruct): weight transform (identity/softmax)

    Return:
    (Float64): the loss

    """

    d_pos = distance(anchor, positive, metric)
    d_neg = distance(anchor, negative, metric)
    
    f(x) = weight_transform(x)
    w = Flux.params(metric) 
    # mutagenesis weigts .... 
    # w = Params([Float32[1.0, 1.0, 1.0, 1.0], Float32[1.0, 1.0, 1.0, 1.0], Float32[1.0, 1.0, 1.0, 1.0, 1.0]])   
    # w = [x₁, x₂, x₃] = [[x₁₁, x₁₂, x₁₃, x₁₄], ..., [x₃₁, x₃₂, x₃₃, x₃₄, x₃₅]]
    # L1 regularization 
    # reg = √(∑ᵢ(∑ⱼ xᵢⱼ²)) ---->   √(x₁₁² + x₁₂² + ... +  x₃₄² + x₃₅²)
    # f(x) is weight transform of each element of weights

    reg = sum(x->sum(abs.(f(x))), w)

    return max(d_pos - d_neg + α, 0) + λₗₐₛₛₒ * reg
end
