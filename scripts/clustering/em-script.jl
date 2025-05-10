### TESTING EM_GMM in detail ###
"""
X, y = generate_data_2d(GaussianData(), 500, 500, 
          [-4.0, 0.0], [4.0, 0.0], [2.0 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0])
plot_classes_2d(X, y, 2)

μ, Σ, γ, y, iters = EM_GMM(X, 2);
plot_distributions_2d(X, μ, Σ, γ)

m = _precision(μ, true_mean)
c = _precision(Σ, hcat(true_cov...))
"""