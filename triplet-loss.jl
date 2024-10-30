using Random, Distances, Flux
using Plots

abstract type SelectingTripletMethod end

struct SelectRandom <: SelectingTripletMethod end
struct SelectHard <: SelectingTripletMethod end

"""
X - n points in an embedding space  
y - labels (length n)
α - margin
λ - learning rate
epochs - number of learning cycles

"""
X = [[1, 2], [3, 2], [5, 2], [7, 2], [9, 2], [2, 4], [4, 4], [6, 4], [8, 4], [10, 4]]
y = [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]
α = 1.0
λ = 0.01
epochs = 20

function separateClasses(X, y, anchor, anchor_label)
    positives = [X[j] for j in 1:length(X) if y[j] == anchor_label && X[j] != anchor]
    negatives = [X[j] for j in 1:length(X) if y[j] != anchor_label]
    return positives, negatives
end

function mahalanobis(x, y, w)
    return sum(w .* ((x .- y) .^ 2))
end

function selectTriplet(::SelectRandom, X, y, w)
    
    i = rand(1:length(X))
    anchor = X[i]
    anchor_label = y[i]
    positives, negatives = separateClasses(X, y, anchor, anchor_label)

    for pos in positives
        d_pos = mahalanobis(anchor, pos, w)
        for neg in negatives
            d_neg = mahalanobis(anchor, neg, w)
            (d_neg > d_pos) && return anchor, pos, neg
        end
    end

    return nothing
end

function selectTriplet(::SelectHard, X, y, w)

    # the first point which satisfies the inequation is returned
    for i in 1:length(X)

        anchor = X[i]
        anchor_label = y[i]
        positives, negatives = separateClasses(X, y, anchor, anchor_label)

        for pos in positives
            d_pos = mahalanobis(anchor, pos, w)
            for neg in negatives
                d_neg = mahalanobis(anchor, neg, w)
                (d_neg < d_pos + α) && return anchor, pos, neg
            end
        end
    end

    return nothing
end

function triplet_loss(anchor, positive, negative, w)
    d_pos = mahalanobis(anchor, positive, w)
    d_neg = mahalanobis(anchor, negative, w)
    return max(d_pos - d_neg + α, 0)
end

function train(method::SelectingTripletMethod)

    w = [1.0, 1.0]
    ps = Flux.params(w)
    opt = Descent(λ)

    for epoch in 1:epochs

        triplet = selectTriplet(method, X, y, w)
        (triplet === nothing) && println("nothing") && continue

        anchor, pos, neg = triplet
        loss_fn = triplet_loss(anchor, pos, neg, w)

        loss, grad = Flux.withgradient(() -> loss_fn, ps)
        Flux.update!(opt, ps, grad)

        println("Epoch $epoch, loss $loss, anchor $anchor, ps $ps")
    end
end

train(SelectRandom())
