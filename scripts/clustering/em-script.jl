### TESTING EM_GMM in detail ###
"""
X, y = generate_data_2d(GaussianData(), 500, 500, 
          [-4.0, 0.0], [4.0, 0.0], [2.0 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0])
plot_classes_2d(X, y, 2)

μ, Σ, γ, y, iters = EM_GMM(Kmeanspp(), X, 2);
plot_distributions_2d(X, μ, Σ, γ)

m = _precision(μ, true_mean)
c = _precision(Σ, hcat(true_cov...))
"""

function main_em_gaussian()

    X, true_labels, true_mean, true_cov = generate_dataset_2d(200, 200);
    μ, Σ, probs, clusters, iters = EM_GMM(init, X, 2)
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
    μ, Σ, probs, clusters, iters = EM_GMM(init, X, 2)

    _rand = randindex(vec(true_labels), vec(y))
    println("\nClustering quality (RI): $(_rand[2])")
    
    #plot_distributions_2d(X, μ, Σ, γ);
    plot_classes_2d(X, y', 2; centroids=μ);
end

function main_em_uniform()

    X, _ = generate_uniform_2d(200, 200);
    μ, Σ, probs, clusters, iters = EM_GMM(init, X, 2)
    plot_distributions_2d(X, μ, Σ, γ);
end

function main_em_laplace()

    X, _ = generate_laplace_2d(200, 200);
    μ, Σ, probs, clusters, iters = EM_GMM(init, X, 2)
    plot_distributions_2d(X, μ, Σ, γ);
end

function test_em(d::DataDistribution, init::InitCenters, n::Int, m::Int, c₁, c₂, v₁, v₂; iter=100)

    X, y = generate_data_2d(d, n, m, c₁, c₂, v₁, v₂)
    average_ri = 0.0
    average_iters = 0.0
    average_m = 0.0
    average_c = 0.0
    for _ in 1:iter
        μ, Σ, probs, clusters, iters = EM_GMM(init, X, 2)
        Σ = hcat(Σ...)
        ri = randindex(vec(y), vec(clusters))[2]
        average_ri = average_ri + ri
        average_iters = average_iters + iters
        # m = min(_precision(μ, hcat(c₁, c₂), 2), (_precision(μ, hcat(c₂, c₁), 2)))
        # c = min(_precision(w, hcat(v₁, v₂), 2), _precision(w, hcat(v₂, v₁), 2))
        # average_m = average_m + m
        # average_c = average_c + c
    end
    println("EM GMM: RI $(average_ri / iter), avrg iterations $(average_iters / iter), mean diff $(average_m / iter), covariance diff $(average_c / iter)")
end