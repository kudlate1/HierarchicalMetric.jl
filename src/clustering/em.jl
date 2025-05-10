function EM_GMM(X, k::Int; max_iter::Int=200)
    """
    Performs EM algorithm for gaussian mixtures, more details in 
    Bishop - Pattern Recognition and Machine Learning, 2006

    Params:
    X (Matrix):     observation points
    k (Int):        the number of clusters
    max_iter (Int): max iterations

    Return:
    μ (Matrix): learned means of the clusters
    Σ (Matrix): learned covariance matrices of the clusters
    π (Vector): learned mixing coefficients
    γ (Matrix): learned posterior probabilities
    """

    d, n = size(X)  # dims, points

    # 1. init μ, Σ, π, log likelihood and γ
    μ = init_centers(Kmeanspp(), X, k)
    Σ = [Matrix(1.0I, d, d) for _ in 1:k]
    π = fill(1/k, k)
    loglike_init = -Inf
    γ = zeros(n, k)
    iters = 0

    for iter in 1:max_iter

        iters = iter

        # 2. E-Step: compute responsibilities 
        compute_responsibilities_gmm(X, μ, Σ, π, γ)

        # 3. M-Step: update parameters
        N = sum(γ, dims=1)[:]
        μ = update_centers(X, γ, N)
        Σ = update_covariances(X, μ, γ, N)
        π = N / n

        # 4. eval log likelihood + convergence check
        log_likelihood = sum(log(sum(π[j] * gaussian(X[:, i], μ[:, j], Σ[j]) for j in 1:k)) for i in 1:n)
        diff = abs(log_likelihood - loglike_init)
        (diff < 1e-4) && break
        loglike_init = log_likelihood
        
        println("Iteration: $iter, log likelihood: $log_likelihood")
    end

    y = classify(γ)

    return μ, Σ, γ, y, iters
end
