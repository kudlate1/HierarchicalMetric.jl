
abstract type SelectingTripletMethod end

struct SelectRandom <: SelectingTripletMethod end
struct SelectHard <: SelectingTripletMethod end


function separateClasses(product_nodes, y, anchor, anchor_label)
    positives = [product_nodes[j] for j in 1:length(product_nodes) if y[j] == anchor_label && product_nodes[j] != anchor]
    negatives = [product_nodes[j] for j in 1:length(product_nodes) if y[j] != anchor_label]
    return positives, negatives
end

distance(pn1, pn2, metric) = only(metric(pn1, pn2))

function myMahalanobis(pn1, pn2, w)
 
    x1 = only(pn1.data.x.data)
    y1 = only(pn1.data.y.data)
    x2 = only(pn2.data.x.data)
    y2 = only(pn2.data.y.data)
    
    return w[1]^2 * (x1 - x2)^2 + w[2]^2 * (y1 - y2)^2
end

function selectTriplet(::SelectRandom, dists, product_nodes, y, metric)
    
    i = rand(1:length(product_nodes))
    anchor = product_nodes[i]
    anchor_label = y[i]

    positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

    positive = rand(positives)
    negative = rand(negatives)

    return anchor, positive, negative
end

function pairwiseDistance(X)

    n = length(X)
    distances = zeros((n, n))
    metric = reflectmetric(X[1], weight_transform=softplus)

    for i in 1:n
        for j in 1:n
            (i != j) && (distances[i, j] = distance(X[i], X[j], metric))
        end
    end
       
    return distances
end

"""
For random k product nodes finds a triplet (the nearest neg and the farthest pos from the dataset)
"""
function selectTriplet(::SelectHard, dists, product_nodes, y, metric)

    n = length(product_nodes)
    k = 20
    perm = Random.randperm(k)
    triplets = []

    for i in perm

        anchor = product_nodes[i]
        anchor_label = y[i]
        positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

        positive = Nothing
        negative = Nothing

        d_pos = 0.0
        d_neg = Inf64

        for pos in 1:n
            if (product_nodes[pos] in positives && dists[i, pos] > d_pos)
                d_pos = dists[i, pos]
                positive = product_nodes[pos]
            end
        end

        for neg in 1:n
            if (product_nodes[neg] in negatives && dists[i, neg] < d_neg)
                d_neg = dists[i, neg]
                negative = product_nodes[neg]
            end
        end

        push!(triplets, (anchor, positive, negative))
    end

    return triplets[rand(1:length(triplets))]
end

function splitData(X, y, numFolds)

    perm = Random.randperm(length(y))
    trn_len = Int64((1 - 1/numFolds) * length(y))

    X_perm = [X[:, i] for i in perm]
    y_perm = [y[i] for i in perm]

    X_trn = reduce(hcat, X_perm[1:trn_len, :])
    y_trn = y_perm[1:trn_len, :]

    X_tst = reduce(hcat, X_perm[trn_len + 1:end, :])
    y_tst = y_perm[trn_len + 1:end, :]

    return X_trn, y_trn, X_tst, y_tst
end

# TODO: which params to crossval
function crossval(X, y, params, numFolds) 

    X_train, y_train, X_test, y_test = splitData(X, y, 5)
    best_λ = Nothing
    bestScore = -Inf64
end

function tripletLoss(anchor, positive, negative, metric; α = 0.1, λₗₐₛₛₒ = 100.0)

    d_pos = distance(anchor, positive, metric)
    d_neg = distance(anchor, negative, metric)
    w = metric.weights.values # Flux.destructure, nebo jinak?

    return max(d_pos - d_neg + α, 0) + λₗₐₛₛₒ * sum(abs.(w)) # lasso nic nedělá
end
