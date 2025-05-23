using HierarchicalMetric

### TESTING LAT in detail ###
"""
X, y = generate_data_2d(GaussianData(), 500, 500, 
          [-4.0, 0.0], [4.0, 0.0], [2.0 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0])

centroids, idxs, y_new, weights, probs, h, losses, iters = LAT_vec(Kmeanspp(), X, 2);

plot_classes_2d(X, y_new, 2)
plot_distributions_2d(X, centroids, weights, probs)
plot_process(weights, h)

X_transformed = weight_transform(X, y_new, weights)
plot_classes_2d(X_transformed, y, 2)
"""

function test_lat(d::DataDistribution, init::InitCenters, n::Int, m::Int, c₁, c₂, v₁, v₂; iter=1000)
    X, y = generate_data_2d(d, n, m, c₁, c₂, v₁, v₂)
    average_ri = 0.0
    average_iters = 0
    average_m = 0.0
    average_c = 0.0
    for _ in 1:iter
        _, clusters, w, _, _, iters = LAT_vec(init, X, 2)
        w = hcat(w...)
        ri = randindex(vec(y), vec(clusters))[2]
        average_ri = average_ri + ri
        average_iters = average_iters + iters
    end
    println("LAT: RI $(average_ri / iter), avrg iterations $(average_iters / iter), mean diff $(average_m / iter), covariance diff $(average_c / iter)")
end

function test_muta_lat()

    X, y = load("data/mutagenesis.json")

    lr = [0.00001, 0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    margin = [0.1, 0.5, 1.0, 2.0, 5.0]
    norm = [0.1, 1.0, 2.0, 5.0, 10.0]
    c1 = []
    pi = []

    for _λ in lr
        for _α in margin
            for _h in norm

                average_ri = 0.0
                average_iters = 0.0
                for _ in 1:10
                    _, _, y_new, _, _, _, iters, π₁ = LAT_htd(Kmeanspp(), X, 2; h=_h, λ=_λ, α=_α);
                    c = sum([1 for i in 1:188 if y_new[i] == 1])
                    push!(c1, c)
                    push!(pi, π₁)
                    ri = randindex(vec(y), vec(y_new))[2]
                    average_ri = average_ri + ri
                    average_iters = average_iters + iters
                end
                println("SETUP: lr=$_λ, RI=$(average_ri / 10), iters=$(average_iters / 10)")
            end
        end
    end

    return c1, pi
end
