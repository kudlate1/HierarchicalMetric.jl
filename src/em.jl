function init_params(d::Int, k::Int)

    Random.seed!(42)
    μ = X[:, rand(1:n, K)]
    Σ = [Matrix(I, d, d) for _ in 1:K]
    π = fill(1/K, K)

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

    d = size(xₙ, 2)
    frac1 = 1 / (2π)^(d/2)
    frac2 = 1 / sqrt(det(Σₖ))
    _exp = -0.5 * (xₙ - μₖ)' * inv(Σₖ) * (xₙ - μₖ)
    return frac1 * frac2 * exp(_exp)
end

function compute_responsibilities(X, π, μ, Σ, γ, n, k)

    for i in 1:n
        total_prob = 0.0
        for j in 1:k
            γ[i, j] = π[j] * gaussian_pdf(X[:, i], μ[:, j], Σ[j])
            total_prob += γ[i, j]
        end
        γ[i, :] ./= total_prob
    end
end

function update_means(X, μ, γ, k)

    for j in 1:k
        μ[:, j] = sum(X .* γ[:, j]', dims=2) / N[j]
    end
end

function update_covariances(X, μ, Σ, γ, N, n, k)

    for j in 1:k
        Σ[j] = zeros(d, d)
        for i in 1:n
            diff = X[:, i] - μ[:, j]
            Σ[j] += γ[i, j] * (diff * diff')
        end
        Σ[j] /= N[j]
    end
end

function EM_GMM(X, k::Int; max_iter::Int=100, atol=1e-4)

    d, n = size(X)  # dims, points

    # 1. init μ, Σ, π, log likelihood and γ
    μ, Σ, π = init_params(d, k)
    loglike_init = -Inf
    γ = zeros(n, k)

    for _ in 1:max_iter

        # 2. E-Step: compute responsibilities 
        γ = compute_responsibilities(X, π, μ, Σ, γ, n, k)

        # 3. M-Step: update parameters
        N = sum(γ, dims=1)[:]
        μ = update_means(X, μ, γ, k)
        Σ = update_covariances(X, μ, Σ, γ, N, n, k)
        π = N / n

        # 4. eval log likelihood + convergence check
        log_likelihood = sum(log(sum(π[k] * gaussian_pdf(X[:, i], μ[:, k], Σ[k]) for k in 1:K)) for i in 1:n)
        abs(log_likelihood - log_likelihood_old) < atol && break
        loglike_init = log_likelihood
    end

    return μ, Σ, π, γ
end