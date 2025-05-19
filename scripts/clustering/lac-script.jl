### TESTING LAC in detail ###
"""
X, y = generate_data_2d(GaussianData(), 500, 500, 
          [-4.0, 0.0], [4.0, 0.0], [2.0 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0])
plot_classes_2d(X, y, 2)

centroids, weights, clusters, iters = LAC(Kmeanspp(), X, 2)
plot_classes_2d(X, clusters, 2; centroids)

X_transformed = weight_transform(X, clusters, y, weights)
plot_classes_2d(X_transformed, clusters, k)

m = means_precision(μ, true_mean)
"""

function test_h(d::DataDistribution, init::InitCenters, n::Int, m::Int, c₁, c₂, v₁, v₂; iter=100)

    X, y = generate_data_2d(d, n, m, c₁, c₂, v₁, v₂)
    _h = [0.01, 0.1, 1.0, 5.0, 10.0, 50.0, 100.0]

    for i in _h
        average_ri = 0.0
        average_iters = 0
        for _ in 1:iter
            _, _, clusters, iters = LAC(init, X, 2; h=i);
            ri = randindex(vec(y), vec(clusters))[2]
            average_ri = average_ri + ri
            average_iters = average_iters + iters
        end
        println("Param h = $i, RI: $(average_ri / iter), avrg iterations: $(average_iters / iter)")
    end
end

function main_lac_gaussian(init::InitCenters)

    X, true_labels, true_means, _ = generate_dataset_2d(200, 200)
    centroids, _, clusters = LAC(init, X, 2)
    println("\nTrue means: $true_means, centroids: $centroids")

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_lac_exponential(init::InitCenters)

    X, true_labels = generate_exponential_2d(200, 200)
    centroids, _, clusters = LAC(init, X, 2)

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_lac_uniform(init::InitCenters)

    X, true_labels = generate_uniform_2d(800, 200);
    centroids, _, clusters = LAC(Kmeanspp(), X, 2)

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_lac_laplace()

    X, true_labels = generate_laplace_2d(200, 200);
    centroids, _, clusters = LAC(Kmeanspp(), X, 2)

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function test_lac(d::DataDistribution, init::InitCenters, n::Int, m::Int, c₁, c₂, v₁, v₂; iter=1000)

    X, y = generate_data_2d(d, n, m, c₁, c₂, v₁, v₂)
    average_ri = 0.0
    average_iters = 0
    for _ in 1:iter
        _, _, clusters, iters = LAC(init, X, 2);
        ri = randindex(vec(y), vec(clusters))[2]
        average_ri = average_ri + ri
        average_iters = average_iters + iters
    end
    println("LAC: RI $(average_ri / iter), avrg iterations $(average_iters / iter)")
end