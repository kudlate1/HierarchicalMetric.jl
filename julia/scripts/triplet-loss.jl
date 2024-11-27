using Random, Distances, Flux
using Plots
using Mill

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

include("../../HSTree/src/metric.jl")

X = Float64.([1 3 5 7 9 2 4 6 8 10; 2 2 2 2 2 3 3 3 3 3])
y = [1 1 1 1 1 0 0 0 0 0]

PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
product_nodes = [PN[i] for i in 1:10]

#pn = product_nodes[1]
#pn.data.x.data |> only

α = 0.1
λ = 0.01
epochs = 200

function plotData(points, labels)

    x_coords = vec([only(point.data.x.data) for point in points])
    y_coords = vec([only(point.data.y.data) for point in points])

    println(x_coords)
    println(y_coords)

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

function separateClasses(product_nodes, y, anchor, anchor_label)
    positives = [product_nodes[j] for j in 1:length(product_nodes) if y[j] == anchor_label && product_nodes[j] != anchor]
    negatives = [product_nodes[j] for j in 1:length(product_nodes) if y[j] != anchor_label]
    return positives, negatives
end

separateClasses(product_nodes, y, product_nodes[5], 1)

function mahalanobis(pn1, pn2, w)
 
    x1 = only(pn1.data.x.data)
    y1 = only(pn1.data.y.data)
    x2 = only(pn2.data.x.data)
    y2 = only(pn2.data.y.data)
    
    return w[1]^2 * (x1 - x2)^2 + w[2]^2 * (y1 - y2)^2
end

function selectTriplet(::SelectRandom, product_nodes, y, w)
    
    i = rand(1:length(product_nodes))
    anchor = product_nodes[i]
    anchor_label = y[i]

    positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

    positive = rand(positives)
    negative = rand(negatives)

    return anchor, positive, negative
end

function selectTriplet(::SelectHard, product_nodes, y, w)

    triplets = []

    for i in 1:length(product_nodes)

        anchor = product_nodes[i]
        anchor_label = y[i]
        positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

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

triplet_loss(product_nodes[1], product_nodes[3], product_nodes[2], [1.0, 1.0])

function train(method::SelectingTripletMethod)

    w = [1.0, 1.0]
    ps = Flux.params(w)
    opt = Descent(λ)

    for epoch in 1:epochs

        triplet = selectTriplet(method, product_nodes, y, w)
        (triplet == nothing) && break 

        anchor, pos, neg = triplet
        (pos == nothing || neg == nothing) && continue

        println(w)

        loss, grad = Flux.withgradient(() -> triplet_loss(anchor, pos, neg, w), ps)
        Flux.update!(opt, ps, grad)

        println("Epoch $epoch, loss $loss, [w1,w2] = $w")
    end

    return w
end

#-----------------------------------------------------------------------------------------------------------

# original dataset
plotData(product_nodes, y)

w = train(SelectRandom())
X_modified = deepcopy(X)

for i in X
    for j in 1:length(X)
        i[j] = i[j] * w[j]
    end
end

# dataset after applying trained parameters w1, w2
plotData(X_modified, y)
