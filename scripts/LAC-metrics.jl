using HierarchicalMetric

function generate_dataset(n::Int)

    function generate_gaussian_data(mean::Vector{Float64}, cov::Matrix{Float64}, n::Int)
        dist = MvNormal(mean, cov)
        return rand(dist, n)'
    end

    mean1 = [-3.0, 0.0]
    cov1 = [1.0 0.0; 0.0 4.0] 
    
    mean2 = [3.0, 0.0]
    cov2 = [4.0 0.0; 0.0 1.0] 
    
    data1 = generate_gaussian_data(mean1, cov1, n)
    data2 = generate_gaussian_data(mean2, cov2, n)

    X = hcat(data1', data2')
    labels = Int.(vcat(fill(1, n), fill(2, n)))

    return X, hcat(labels...)
end

function weight_transform(X::Matrix, y::Matrix, y_true::Matrix, weights::Matrix)

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

function plot_classes(X, y, k; centroids=Nothing)

    p = plot()
    colors = [:red, :blue, :yellow, :green, :orange, :purple, :cyan, :magenta, :brown, :pink]

    if centroids != Nothing
        k = k + 1
        X = hcat(X, centroids)
        y = hcat(y, ones(1, k-1) * k)
    end

    for class in 1:k 
        class_points = X[:, vec(y) .== class]
        scatter!(
            p, 
            background_color = RGB(0.4, 0.4, 0.4), 
            class_points[1, :], 
            class_points[2, :], 
            label="Class $class",
            marker = (5, :circle),
            color=colors[class],
            xlims=(-10, 10),
            ylims=(-10, 10)
        )
    end

    display(p)
end

"""
X, true_labels = generate_dataset(100)
plot_classes(X, true_labels, 2)

centroids, weights, clusters = LAC(X, 2)
plot_classes(X, clusters, 2; centroids)

X_transformed = weight_transform(X, clusters, true_labels, weights)
plot_classes(X_transformed, clusters, k)

vm, var, silh, rand = metrics(X, vec(true_labels), vec(clusters))

"""