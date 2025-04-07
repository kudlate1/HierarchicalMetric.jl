function generate_dataset_2d(n::Int, m::Int)

    function generate_gaussian_data(mean::Vector{Float64}, cov::Matrix{Float64}, n::Int)
        dist = MvNormal(mean, cov)
        return rand(dist, n)'
    end

    mean1 = [-4.0, 0.0]
    cov1 = [1.0 0.0; 0.0 5.0] 
    
    mean2 = [4.0, 0.0]
    cov2 = [5.0 0.0; 0.0 1.0]  
    
    data1 = generate_gaussian_data(mean1, cov1, n);
    data2 = generate_gaussian_data(mean2, cov2, m);

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...), hcat(mean1, mean2), [cov1, cov2]
end

function generate_exponential_2d(n::Int, m::Int)

    function generate_exponential_data(rate::Vector{Float64}, shift::Vector{Float64}, n::Int)
        d1 = Exponential(rate[1])
        d2 = Exponential(rate[2])
        x = rand(d1, n)
        y = rand(d2, n)
        return [x .+ shift[1] y .+ shift[2]]
    end

    rate1 = [4.0, 1.0]
    shift1 = [-6.0, -6.0]

    rate2 = [4.25, 1.0]
    shift2 = [0.0, 0.0]

    data1 = generate_exponential_data(rate1, shift1, n)
    data2 = generate_exponential_data(rate2, shift2, m)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end

function generate_uniform_2d(n::Int, m::Int)

    function generate_uniform_data(bounds::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}, n::Int)
        x = rand(Uniform(bounds[1]...), n)
        y = rand(Uniform(bounds[2]...), n)
        return [x y]
    end

    bounds1 = ((-5.0, -1.0), (-2.0, 2.0))
    bounds2 = ((0.0, 1.0), (-2.0, 2.0))
    
    data1 = generate_uniform_data(bounds1, n)
    data2 = generate_uniform_data(bounds2, m)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end

function generate_laplace_2d(n::Int, m::Int)

    function generate_laplace_data(mean::Vector{Float64}, scale::Vector{Float64}, n::Int)
        d1 = Laplace(mean[1], scale[1])
        d2 = Laplace(mean[2], scale[2])
        x = rand(d1, n)
        y = rand(d2, n)
        return [x y]
    end

    mean1 = [-4.0, 0.0]
    scale1 = [1.0, 2.0]
    
    mean2 = [4.0, 0.0]
    scale2 = [2.0, 1.0] 
    
    data1 = generate_laplace_data(mean1, scale1, n)
    data2 = generate_laplace_data(mean2, scale2, m)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end