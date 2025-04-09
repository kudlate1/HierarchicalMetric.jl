using HierarchicalMetric

function weight_transform(X::Matrix, y::Matrix, weights::Matrix)

    X = [x .* weights[:, y[x_i]] for (x_i, x) in enumerate(eachcol(X))]
    return hcat(X...)
end

function metrics(X, true_labels, clusters)

    vm = vmeasure(true_labels, clusters)
    var = varinfo(true_labels, clusters)

    # nefunguje...
    # distances = pairwise(Euclidean(), X')
    # silh = silhouettes(distances, Float64.(true_labels))
    silh = Nothing

    rand = randindex(true_labels, clusters)

    return vm, var, silh, rand
end

squared_distance(x, y) = sum((x .- y).^2)

means_precision(found_mean, true_mean) = squared_distance(found_mean, true_mean)

covariances_precision(found_cov, true_cov, dims) = sum(squared_distance.(found_cov, true_cov)) / dims

function main_lac_gaussian()

    X, true_labels, true_means, _ = generate_dataset_2d(200, 200)
    centroids, _, clusters = LAC(X, 2)
    println("\nTrue means: $true_means, centroids: $centroids")

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_lac_exponential()

    X, true_labels = generate_exponential_2d(200, 200)
    centroids, _, clusters = LAC(X, 2)

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_lac_uniform()

    X, true_labels = generate_uniform_2d(800, 200);
    centroids, _, clusters = LAC(X, 2)

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_lac_laplace()

    X, true_labels = generate_laplace_2d(200, 200);
    centroids, _, clusters = LAC(X, 2)

    _rand = randindex(vec(true_labels), vec(clusters))
    println("\nClustering quality (RI): $(_rand[2])")

    plot_classes_2d(X, clusters, 2; centroids);
end

function main_em_gaussian()

    X, true_labels, true_mean, true_cov = generate_dataset_2d(200, 200);
    μ, Σ, γ, y = EM_GMM(X, 2);
    println("\nTrue means: $true_mean, learned means: $μ")
    println("True covariances: $true_cov, learned covariances: $Σ")

    m = means_precision(μ, true_mean);
    c = covariances_precision(Σ, true_cov, 2);

    println("\nMean difference: $m")
    println("Covariance difference: $c")

    _rand = randindex(vec(true_labels), vec(y))
    println("\nClustering quality (RI): $(_rand[2])")

    #plot_distributions_2d(X, μ, Σ, γ);
    plot_classes_2d(X, y', 2; centroids=μ)
end

function main_em_exponential()

    X, true_labels = generate_exponential_2d(200, 200);
    μ, Σ, γ, y = EM_GMM(X, 2);

    _rand = randindex(vec(true_labels), vec(y))
    println("\nClustering quality (RI): $(_rand[2])")
    
    #plot_distributions_2d(X, μ, Σ, γ);
    plot_classes_2d(X, y', 2; centroids=μ);
end

function main_em_uniform()

    X, _ = generate_uniform_2d(200, 200);
    μ, Σ, γ = EM_GMM(X, 2);
    plot_distributions_2d(X, μ, Σ, γ);
end

function main_em_laplace()

    X, _ = generate_laplace_2d(200, 200);
    μ, Σ, γ = EM_GMM(X, 2);
    plot_distributions_2d(X, μ, Σ, γ);
end


### TESTING LAC in detail ###
"""
X, true_labels, _, _ = generate_dataset_2d(100)
plot_classes_2d(X, true_labels, 2)

centroids, weights, clusters = LAC(X, 2)
plot_classes_2d(X, clusters, 2; centroids)

X_transformed = weight_transform(X, clusters, true_labels, weights)
plot_classes_2d(X_transformed, clusters, k)

m = means_precision(μ, true_mean)
c = covariances_precision(Σ, true_cov)

rand = randindex(vec(true_labels), vec(clusters))
"""

### TESTING EM_GMM in detail ###
"""
X, true_labels, true_mean, true_cov = generate_dataset_2d(100);
plot_classes_2d(X, true_labels, 2)

μ, Σ, γ = EM_GMM(X, 2);
plot_distributions_2d(X, μ, Σ, γ)

m = means_precision(μ, true_mean)
c = covariances_precision(Σ, true_cov)
"""
