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
    labels = vcat(fill(1, n), fill(2, n))

    return X, labels
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

function plot_classes(X, y, k)

    p = plot()
    colors = [:red, :blue, :yellow, :green, :orange, :purple, :cyan, :magenta, :brown, :pink]

    for class in 1:k
        class_points = X[:, vec(y) .== class]
        scatter!(
            p, 
            background_color = RGB(0.4, 0.4, 0.4), 
            class_points[1, :], 
            class_points[2, :], 
            label="Class $class",
            marker = (5, :circle), 
            color=colors[class]
        )
    end

    display(p)
end

"""
k = 2
X, true_labels = generate_dataset()
plotData(X, true_labels)

centroids, weights, clusters = LAC(X, k)

X2 = hcat(X, centroids)
y2 = hcat(clusters, ones(1, k) * (k+1))
plot_classes(X2, y2, k+1)

vm, var, silh, rand = metrics(X, Int.(vec(true_labels)), Int.(vec(clusters)))

"""