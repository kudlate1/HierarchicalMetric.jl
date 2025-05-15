function plot_classes_2d(X, y, k; centroids=Nothing)

    p = plot()
    colors = [:red, :blue, :yellow, :green, :orange, :purple, :cyan]
    (typeof(y) <: Vector) && (y = hcat(y...))

    if centroids != Nothing
        k = k + 1
        X = hcat(X, centroids)
        y = hcat(y, ones(1, k-1) * k)
    end

    for class in 1:k 
        class_points = X[:, vec(y) .== class]
        scatter!(
            p, 
            class_points[1, :], 
            class_points[2, :],
            label="Class $class",
            marker = (5, :circle),
            color=colors[class]
        )
    end

    #title!("Labels of the LAT, phase $iter")

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
        background_color = RGB(0.4, 0.4, 0.4), 
        marker_z=max_resp,
        markersize=5,
        legend=false
    )
    
    for j in 1:k
        plot_gaussian_ellipse(μ[:, j], Σ[j], colors[j]);
    end

    title!("Soft assignments of the GMM")

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

    p = plot(reduce(hcat, h)', 
        xlabel="number of iteations", 
        ylabel="values of the parameters", 
        title="Parameters learning with lasso regularization"
    )
    display(p)

    return vcat(softplus.(ps)[1])
end
