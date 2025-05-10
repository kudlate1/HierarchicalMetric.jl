### TESTING LAC in detail ###
"""
X, y = generate_data_2d(GaussianData(), 500, 500, 
          [-4.0, 0.0], [4.0, 0.0], [2.0 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0])
plot_classes_2d(X, y, 2)

centroids, weights, clusters = LAC(X, 2)
plot_classes_2d(X, clusters, 2; centroids)

X_transformed = weight_transform(X, clusters, y, weights)
plot_classes_2d(X_transformed, clusters, k)

m = means_precision(μ, true_mean)
"""