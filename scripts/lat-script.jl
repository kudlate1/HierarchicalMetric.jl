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
        c, clusters, w, probs, h, iters = LAT_vec(init, X, 2)
        w = hcat(w...)
        ri = randindex(vec(y), vec(clusters))[2]
        average_ri = average_ri + ri
        average_iters = average_iters + iters
        # m = min(_precision(c, hcat(c₁, c₂), 2), _precision(c, hcat(c₂, c₁), 2))
        # c = min(_precision(w, hcat(v₁, v₂), 2), _precision(w, hcat(v₂, v₁), 2))
        # average_m = average_m + m
        # average_c = average_c + c
    end
    println("LAT: RI $(average_ri / iter), avrg iterations $(average_iters / iter), mean diff $(average_m / iter), covariance diff $(average_c / iter)")
end


