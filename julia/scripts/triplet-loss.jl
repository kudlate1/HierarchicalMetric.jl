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
X = [[1.0, 2.0], [3.0, 2.0], [5.0, 2.0], [7.0, 2.0], [9.0, 2.0], 
     [2.0, 3.0], [4.0, 3.0], [6.0, 3.0], [8.0, 3.0], [10.0, 3.0]]
y = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
α = 0.1
λ = 0.9
epochs = 200

function plotData(points, labels)

    x_coords = [point[1] for point in points]
    y_coords = [point[2] for point in points]

    colors = [label == 1 ? RGB(0.2, 0.2, 0.8) : RGB(0.4, 0.1, 0.2) for label in labels]

    scatter(
        x_coords,
        y_coords,
        group = labels,
        color = colors,
        marker = (10, :circle),
        xlabel = "x",
        ylabel = "y",
        background_color = RGB(0.2, 0.2, 0.2),
        legend = false
    )
end

function separateClasses(X, y, anchor, anchor_label)
    positives = [X[j] for j in 1:length(X) if y[j] == anchor_label && X[j] != anchor]
    negatives = [X[j] for j in 1:length(X) if y[j] != anchor_label]
    return positives, negatives
end

function mahalanobis(x, y, w)
    return sum(w[i] * ((x[i] - y[i])^2) for i in 1:length(x))
end

function selectTriplet(::SelectRandom, X, y, w)
    
    i = rand(1:length(X))
    anchor = X[i]
    anchor_label = y[i]

    positives, negatives = separateClasses(X, y, anchor, anchor_label)

    positive = rand(positives)
    negative = rand(negatives)

    return anchor, positive, negative
end

function selectTriplet(::SelectHard, X, y, w)

    triplets = []

    for i in 1:length(X)

        anchor = X[i]
        anchor_label = y[i]
        positives, negatives = separateClasses(X, y, anchor, anchor_label)

        for pos in positives
            d_pos = mahalanobis(anchor, pos, w)

            for neg in negatives
                d_neg = mahalanobis(anchor, neg, w)

                if d_neg < d_pos + α
                    push!(triplets, (anchor, pos, neg))
                end
            end
        end
    end

    ret = (length(triplets) == 0) ? nothing : triplets[rand(1:length(triplets))]
    return ret
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
        (triplet == nothing) && break 

        anchor, pos, neg = triplet
        (pos == nothing || neg == nothing) && continue

        loss, grad = Flux.withgradient(() -> triplet_loss(anchor, pos, neg, w), ps)
        Flux.update!(opt, ps, grad)

        println("Epoch $epoch, loss $loss, triplet [$anchor, $pos, $neg], [w1,w2] = $w")
    end

    return w
end

# original dataset
plotData(X, y)

w = train(SelectHard())
X_modified = deepcopy(X)

for i in 1:length(X)
    for j in 1:2
        X_modified[i][j] = X[i][j] * w[j]
    end
end

# dataset after applying trained parameters w1, w2
plotData(X_modified, y)

