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

function test_em(d::DataDistribution, init::InitCenters, n::Int, m::Int, c₁, c₂, v₁, v₂; iter=100)

    X, y = generate_data_2d(d, n, m, c₁, c₂, v₁, v₂)
    average_ri = 0.0
    average_iters = 0.0
    average_m = 0.0
    average_c = 0.0
    for _ in 1:iter
        μ, Σ, _, clusters, iters = EM_GMM(init, X, 2)
        Σ = hcat(Σ...)
        ri = randindex(vec(y), vec(clusters))[2]
        average_ri = average_ri + ri
        average_iters = average_iters + iters
        m = min(_precision(μ, hcat(c₁, c₂), 2), (_precision(μ, hcat(c₂, c₁), 2)))
        c = min(_precision(w, hcat(v₁, v₂), 2), _precision(w, hcat(v₂, v₁), 2))
        average_m = average_m + m
        average_c = average_c + c
    end
    println("EM GMM: RI $(average_ri / iter), avrg iterations $(average_iters / iter), mean diff $(average_m / iter), covariance diff $(average_c / iter)")
end