function plot_classes_2d(X, y, k; centroids=Nothing)

    p = plot()
    colors = [:red, :blue, :yellow, :green, :orange, :purple, :cyan]
    (typeof(y) <: Vector) && (y = hcat(y...))

    for class in 1:k 
        class_points = X[:, vec(y) .== class]
        scatter!(
            p, 
            class_points[1, :], 
            class_points[2, :],
            label="class $class",
            marker = (5, :circle),
            color=colors[class]
        )
    end

    if centroids != Nothing
        scatter!(
            p,
            centroids[1, :],
            centroids[2, :],
            label="centroids",
            marker = (5, :circle),
            color=colors[3]
        )
    end

    #title!("Title")

    display(p)
end

function plot_distributions_2d(X, μ, Σ, γ)

    function plot_gaussian_ellipse(μ, Σ, color)
        θ = range(0, 2π, length=100)
        vals, vecs = eigen(Σ)
        r = sqrt.(vals)
        ellipse = [r[1] * cos.(θ) r[2] * sin.(θ)] * vecs' .+ μ'
        plot!(ellipse[:, 1], ellipse[:, 2], color=color, linewidth=2)
    end

    p = plot()

    k = size(γ, 2)
    assignments = [argmax(γᵢ) for γᵢ in eachrow(γ)]
    max_resp = maximum(γ, dims=2)[:]
    colors = [:red, :blue, :green, :yellow, :orange, :purple, :cyan]

    scatter!(
        p,
        X[1, :], 
        X[2, :], 
        c=assignments, 
        marker_z=max_resp,
        markersize=5,
        legend=false
    )
    
    for j in 1:k
        plot_gaussian_ellipse(μ[:, j], Σ[j], colors[j]);
    end

    title!("Soft assignments of the LAT")

    return p
end

function plot_data(data, labels)

    x_coords = vec([i[1] for i in eachcol(data)])
    y_coords = vec([i[2] for i in eachcol(data)])

    colors = [label == 1 ? "red" : "blue" for label in vec(labels)]

    scatter(
        x_coords,
        y_coords,
        color = colors,
        xlabel = "x",
        ylabel = "y",
        background_color = RGB(0.4, 0.4, 0.4),
        markersize=5,
        xlims=(-20.0, 20.0),
        ylims=(-20.0, 20.0),
        legend = false
    )

    title!("Separable dataset")
end

function plot_process(ps, h)

    data = reduce(hcat, h)'
    d = size(data, 2)
    n = size(data, 1)

    p = plot()
    for i in 1:d
        plot!(1:n, data[:, i], label="w$i")
    end

    xlabel!("number of iterations")
    ylabel!("values of the parameters")
    title!("Parameters learning with lasso regularization")
    display(p)

    return vcat(softplus.(ps)[1])
end

function plot_ratio(class_counts::Vector, lr_vals::Vector, π::Vector)

    bins = length(class_counts)
    x_labels = string.(lr_vals)
    base_height = 188
    bar_heights = fill(base_height, bins)
    pi = π .* base_height

    bar1 = bar(1:bins, 
               bar_heights, 
               title="Ratio of the classes",
               xlabel="value of λ",
               ylabel="ratio of the classes",
               xticks = (1:10, x_labels), 
               color=:lightblue)
    bar!(1:10, 
         class_counts, 
         color=:darkblue,
         yticks=(0:10:base_height))

    for i in 1:bins
        annotate!(i, class_counts[i] + 5, text(base_height - class_counts[i], :black, 8))
        annotate!(i, class_counts[i] - 5, text(class_counts[i], :white, 8))
    end

    plot!(pi, seriestype = :line, color=:red, label="π")

    display(bar1)
end

function visualise_distances(distances)
    heatmap(distances, aspect_ratio = 1)
end
