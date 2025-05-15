function LAC(init::InitCenters, X::Matrix, k::Int; max_iter::Int=50, h::Float64=10.0)
    """
    Performs LAC algorithm.

    Params:
    X (Matrix):         points
    c (Matrix):         initial centroids
    k (Int):            number of clusters (centroids respectively)
    d (Int):            number of dimensions
    max_iter (Int):     max number of iterations
    h (Float64)

    Return:
    (Matrix): trained centroids
    """

    d, n = size(X) # dims, points

    # 1., 2. init centroids, weights and labels
    c = init_centers(init, X, k)
    #centroids = X[:, rand(1:n, k)] # rnd points
    #centroids = farthest_point(X, k)
    w = 1/d * ones(d, k)
    y = zeros(Int, n)
    iters = 0

    for iter in 1:max_iter

        iters = iter
        last_weights = w
        last_centroids = c

        # 3. sort points into the closest clusters
        for (l, x_l) in enumerate(eachcol(X))
            distances = [Lw(c[:, i], x_l, w[:, i]) for i in 1:k]
            j = argmin(distances)
            y[l] = j
        end

        # 4. compute new weights
        w = set_weights_lac(X, c, y, h)
        
        # 5. resort points into clusters Sⱼ
        for (l, x_l) in enumerate(eachcol(X))
            distances = [Lw(c[:, i], x_l, w[:, i]) for i in 1:k]
            j = argmin(distances)
            y[l] = j
        end

        # 6. recompute centroids
        c = set_centroids_lac(X, y, k)

        # 7. convergence check
        weight_diff = sum((w .- last_weights).^2)
        centroid_diff = sum((c .- last_centroids).^2)

        #println("Iteration $iter: weight change = $weight_diff, centroid change = $centroid_diff")
        (weight_diff <= 1e-4 && centroid_diff <= 1e-4) && break
    end

    return c, w, y, iters
end
