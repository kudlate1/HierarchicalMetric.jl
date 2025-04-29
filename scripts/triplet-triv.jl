using HierarchicalMetric
using HierarchicalMetric.Mill
using Plots

"""
Example of Mill.jl functions 

Values in the ProductNode access:
pn = product_nodes[1]
pn.data.x.data |> only  # elementary data type (int, str,...)

Creating the metric using ProductMetric:
lx = LeafMetric(Pairwise_SqEuclidean, "con", "x")
ly = LeafMetric(Pairwise_SqEuclidean, "con", "y")
PM = ProductMetric((x = lx, y = ly), SqWeightedProductMetric, WeightStruct((x = 1f0, y = 1f0), softplus), "name")

Creating the metric using reflectmetric (equivivalent):
PM2 = reflectmetric(product_nodes[1])

Computing the distance of two ProductNodes using metric (PM/PM2):
only(PM2(product_nodes[1], product_nodes[2]))
"""

function create_product_nodes(data)

    _, n = size(data)
    PN = ProductNode((x = Array(data[1, :]'), y  = Array(data[2, :]')))
    X = [PN[i] for i in 1:n]
    return X
end

function visualise_distances(distances)
    heatmap(distances, aspect_ratio = 1)
end

function test_triplet(max_iter::Int)

    λ = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    data, y = generate_separable_2d(GaussianData(), 50, 50, [-1.0, 0.0], [1.0, 0.0], [0.1 0.0; 0.0 10.0], [0.1 0.0; 0.0 10.0])
    X = create_product_nodes(data)
    distances = pairwise_distance(X)

    for l in λ
        avrg_iters = 0.0
        avrg_w1 = 0.0
        avrg_w2 = 0.0
        for _ in 1:max_iter
            ps, _, iters = train(SelectRandom(), X, y; λ=l);
            avrg_iters = avrg_iters + iters
            avrg_w1 = avrg_w1 + ps[1]
            avrg_w2 = avrg_w2 + ps[2]
        end
        println("Lambda λ = $l: average iterations $(avrg_iters / max_iter), average w₁ $(avrg_w1 / max_iter), average w₂ $(avrg_w2 / max_iter)")
    end
end

### EXAMPLE OF A TRAINING EXECUTION ON TRIVIAL DATASET ###
"""
data, y = generate_separable_2d(GaussianData(), 50, 50, 
          [-4.0, 0.0], [4.0, 0.0], [1.0 1.0; 1.0 5.0], [5.0 1.0; 1.0 1.0])

X = create_product_nodes(data)

distances = pairwise_distance(X)
visualise_distances(distances)

ps, h = train(SelectRandom(), X, y);
X2 = [x .* ps for x in eachcol(data)]
X3 = hcat(X2...)
plot_classes_2d(X3, y, 2)

"""
