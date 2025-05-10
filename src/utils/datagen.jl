function generate_data_2d(::GaussianData, n::Int, m::Int, c₁::Vector, c₂::Vector, v₁::Matrix, v₂::Matrix)
    """
    Generates two classes with Gaussian (normal) distribution. More about the
    Gaussian's parameters in 2d on: https://www.youtube.com/watch?v=UVvuwv-ne1I

    Params:
    n, m (Int): number of points in each class
    c₁, c₂ (Vector): means (centers) of the distributions
    v₁, v₂ (Matrix): covariance matrices

    Return:
    Matrix with point and their ground labels

    An example of the input params:
    mean1 = [-4.0, 0.0]
    cov1 = [1.0 0.0; 0.0 5.0] 
    
    mean2 = [4.0, 0.0]
    cov2 = [5.0 0.0; 0.0 1.0]
    """

    function generate_gaussian_data(mean::Vector{Float64}, cov::Matrix{Float64}, n::Int)
        dist = MvNormal(mean, cov)
        return rand(dist, n)'
    end  
    
    data1 = generate_gaussian_data(c₁, v₁, n);
    data2 = generate_gaussian_data(c₂, v₂, m);

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end

function generate_data_2d(::ExponentialData, n::Int, m::Int, c₁::Vector, c₂::Vector, v₁::Vector, v₂::Vector)
    """
    Generates two classes with exponential distribution. More about the
    parameters of the distribution on: https://en.wikipedia.org/wiki/Exponential_distribution

    Params:
    n, m (Int): number of points in each class
    c₁, c₂ (Vector): shifts (centers) of the distributions
    v₁, v₂ (Vector): the parameters λ (each vᵢ has two params λ for each dimension)

    Return:
    Matrix with point and their ground labels

    An example of the input params:
    rate1 = [4.0, 1.0]
    shift1 = [6.0, 6.0]

    rate2 = [4.25, 1.0]
    shift2 = [0.0, 0.0]
    """

    function generate_exponential_data(rate::Vector{Float64}, shift::Vector{Float64}, n::Int)
        d1 = Exponential(rate[1])
        d2 = Exponential(rate[2])
        x = rand(d1, n)
        y = rand(d2, n)
        return [x .+ shift[1] y .+ shift[2]]
    end

    data1 = generate_exponential_data(c₁, v₁, n)
    data2 = generate_exponential_data(c₂, v₂, m)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end

function generate_data_2d(::UniformData, n::Int, m::Int, c₁::Tuple, c₂::Tuple, v₁::Tuple, v₂::Tuple)
    """
    Generates two classes with uniform distribution. More about the
    parameters of the distribution on: https://en.wikipedia.org/wiki/Continuous_uniform_distribution

    Params:
    n, m (Int): number of points in each class
    c₁, c₂ (::Tuple): the bounds of the first class (c₁ for x-axis, c₂ for y-axis)
    v₁, v₂ (::Tuple): the bounds of the second class (v₁ for x-axis, v₂ for y-axis)

    Return:
    Matrix with point and their ground labels

    An example of the input params:
    bounds_x_c1 = (-5.0, -1.0)
    bounds_y_c1 = (-2.0, 2.0)

    bounds_x_c2 = (0.0, 1.0)
    bounds_y_c2 = (-2.0, 2.0)
    """

    function generate_uniform_data(bounds::Tuple{Tuple{Float64, Float64}, Tuple{Float64, Float64}}, n::Int)
        x = rand(Uniform(bounds[1]...), n)
        y = rand(Uniform(bounds[2]...), n)
        return [x y]
    end
    
    data1 = generate_uniform_data((c₁, c₂), n)
    data2 = generate_uniform_data((v₁, v₂), m)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end

function generate_data_2d(::LaplaceData, n::Int, m::Int, c₁::Vector, c₂::Vector, v₁::Vector, v₂::Vector)
    """
    Generates two classes with Laplace distribution. More about the
    parameters of the distribution on: https://en.wikipedia.org/wiki/Laplace_distribution

    Params:
    n, m (Int): number of points in each class
    c₁, c₂ (Vector): means (centers) of the distributions
    v₁, v₂ (Vector): scales

    Return:
    Matrix with point and their ground labels

    An example of the input params:
    mean1 = [-4.0, 0.0]
    scale1 = [1.0, 2.0]
    
    mean2 = [4.0, 0.0]
    scale2 = [2.0, 1.0] 
    """

    function generate_laplace_data(mean::Vector{Float64}, scale::Vector{Float64}, n::Int)
        d1 = Laplace(mean[1], scale[1])
        d2 = Laplace(mean[2], scale[2])
        x = rand(d1, n)
        y = rand(d2, n)
        return [x y]
    end
    
    data1 = generate_laplace_data(c₁, v₁, n)
    data2 = generate_laplace_data(c₂, v₂, m)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, m)))

    return X, hcat(labels...)
end

function is_separable(X, y)
    
    _, n = size(X)
    class1 = [X[:, j] for j in 1:n if y[j] == 1]
    class2 = [X[:, j] for j in 1:n if y[j] == 2]

    class1 = hcat(class1...)
    class2 = hcat(class2...)

    c1_x_min, c1_x_max = minimum(class1[1, :]), maximum(class1[1, :])
    c1_y_min, c1_y_max = minimum(class1[2, :]), maximum(class1[2, :])
    
    c2_x_min, c2_x_max = minimum(class2[1, :]), maximum(class2[1, :])
    c2_y_min, c2_y_max = minimum(class2[2, :]), maximum(class2[2, :])

    vertical = c1_x_max < c2_x_min || c2_x_max < c1_x_min
    horizontal = c1_y_max < c2_y_min || c2_y_max < c1_y_min

    return vertical || horizontal
end

function generate_separable_2d(d::DataDistribution, n::Int, m::Int, c₁, c₂, v₁, v₂)

    """
    Generates the wantedtype distribution, which is also vertically or horizontally separable. 
    Useful for triplet loss testing. Params corespond to the particular distriution.
    """

    if d isa GaussianData
        println("Handling Gaussian distribution")
        X, y = generate_data_2d(GaussianData(), n, m, c₁, c₂, v₁, v₂)
        plot_classes_2d(X, y, 2)
        !(is_separable(X, y)) && error("Data not separable! Estrange the means, or change the covariances.")

    elseif d isa ExponentialData
        println("Handling Exponential distribution")
        X, y = generate_data_2d(ExponentialData(), n, m, c₁, c₂, v₁, v₂)
        plot_classes_2d(X, y, 2)
        !(is_separable(X, y)) && error("Data not separable! Try to more estrange the shifts, or shrink the rates.")

    elseif d isa UniformData
        println("Handling Uniform distribution")
        X, y = generate_data_2d(UniformData(), n, m, c₁, c₂, v₁, v₂)
        plot_classes_2d(X, y, 2)
        !(is_separable(X, y)) && error("Data not separable! Change the bounds and estrange the classes.")

    elseif d isa LaplaceData
        println("Handling Laplace distribution")
        X, y = generate_data_2d(LaplaceData(), n, m, c₁, c₂, v₁, v₂)
        plot_classes_2d(X, y, 2)
        !(is_separable(X, y)) && error("Data not separable! Try to more estrange the means, or change the scale.")

    else
        error("Unknown distribution!")
    end

    return X, y
end

# todo: do budoucna vic osetrit vstupy