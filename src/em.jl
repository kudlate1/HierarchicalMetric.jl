function init_params(X, k::Int)

    d, n = size(X)

    #μ = X[:, rand(1:n, k)] # rnd points
    μ = kmeanspp(X, k)
    Σ = [Matrix(1.0I, d, d) for _ in 1:k]
    π = fill(1/k, k)

    return μ, Σ, π
end

function gaussian(xₙ, μₖ, Σₖ)
    """
    Computes the probability density for a particular xₙ, μₖ and Σₖ.

    xₙ: n-th observation point
    μₖ: the mean vector of cluster k
    Σₖ: the covariance matrix for cluster k

    Return:
    xₙ, μₖ, Σₖ: initialized parameters
    """

    d = size(xₙ, 1)

    frac1 = 1 / (2π)^(d/2)
    frac2 = 1 / sqrt(det(Σₖ))
    _exp = -0.5 * (xₙ - μₖ)' * inv(Σₖ) * (xₙ - μₖ)

    return frac1 * frac2 * exp(_exp)
end

function compute_responsibilities(X, μ, Σ, π, γ)

    n = size(X, 2)
    k = size(μ, 2)

    for i in 1:n
        total_prob = 0.0
        for j in 1:k
            γ[i, j] = π[j] * gaussian(X[:, i], μ[:, j], Σ[j])
            total_prob += γ[i, j]
        end
        γ[i, :] ./= total_prob
    end
end

function update_means(X, μ, γ, N)

    k = size(μ, 2)

    for j in 1:k
        μ[:, j] = sum(X .* γ[:, j]', dims=2) / N[j]
    end

    return μ
end

function update_covariances(X, μ, Σ, γ, N)

    d, n = size(X)
    k = size(μ, 2)

    for j in 1:k
        Σ[j] = zeros(d, d)
        for i in 1:n
            diff = X[:, i] - μ[:, j]
            Σ[j] += γ[i, j] * (diff * diff')
        end
        Σ[j] /= N[j]
    end

    return Σ
end

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

    _, n = size(X)  # dims, points

    # 1. init μ, Σ, π, log likelihood and γ
    μ, Σ, π = init_params(X, k)
    loglike_init = -Inf
    γ = zeros(n, k)

    for iter in 1:max_iter

        # 2. E-Step: compute responsibilities 
        compute_responsibilities(X, μ, Σ, π, γ)

        # 3. M-Step: update parameters
        N = sum(γ, dims=1)[:]
        μ = update_means(X, μ, γ, N)
        Σ = update_covariances(X, μ, Σ, γ, N)
        π = N / n

        # 4. eval log likelihood + convergence check
        log_likelihood = sum(log(sum(π[j] * gaussian(X[:, i], μ[:, j], Σ[j]) for j in 1:k)) for i in 1:n)
        # logsumexp()
        diff = abs(log_likelihood - loglike_init)
        (diff < 1e-4) && break
        loglike_init = log_likelihood
        println("Iteration: $iter, log likelihood diff: $diff")
    end



    return μ, Σ, γ
end