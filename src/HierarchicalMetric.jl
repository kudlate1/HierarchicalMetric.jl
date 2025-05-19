module HierarchicalMetric

using Pkg
using Revise
using Random, Distances
using Flux
using Mill
using HSTreeDistance
using JsonGrinder, JSON3
using Plots, Colors
using Statistics, Distributions
using Clustering
using LinearAlgebra

# --------------------------------------------------------------

# structure for triplet selection
abstract type TripletSelectionMethod end
struct SelectRandom <: TripletSelectionMethod end
struct SelectHard <: TripletSelectionMethod end
export TripletSelectionMethod, SelectHard, SelectRandom

# structure for generated data distributions
abstract type DataDistribution end
struct GaussianData <: DataDistribution end
struct ExponentialData <: DataDistribution end 
struct UniformData <: DataDistribution end
struct LaplaceData <: DataDistribution end
export DataDistribution, GaussianData, ExponentialData, UniformData, LaplaceData

# structure for centers initialization
abstract type InitCenters end
struct Kmeanspp <: InitCenters end
struct FarthestPoint <: InitCenters end
struct RandomPoint <: InitCenters end
export Kmeanspp, FarthestPoint, RandomPoint, InitCenters

# --------------------------------------------------------------

include("tripletloss/triplet-loss.jl")
export pairwise_distance
export pairwise_distance_htd
export select_triplet_htd
export triplet_loss_htd
export _mahalanobis_pn
export htd
export _mahalanobis_vec
export _mahalanobis_mtx
export select_triplet_vec
export triplet_loss_vec

include("tripletloss/train-tl-htd.jl")
export train_tl_htd

# --------------------------------------------------------------

include("clustering/cluster-utils.jl")
export set_centroids_lac
export set_weights_lac
export Lw
export indicator
export gaussian
export compute_responsibilities_gmm
export update_centers
export update_covariances
export classify
export compute_responsibilities_lat
export compute_responsibilities_htd
export init_centers
export metrics
export weight_transform
export _precision

include("clustering/lac.jl")
export LAC

include("clustering/em.jl")
export EM_GMM

include("clustering/lat.jl")
export LAT_vec
export LAT_htd

# --------------------------------------------------------------

include("../scripts/tripletloss/triplet-triv.jl")
export create_product_nodes
export visualise_distances
export test_triplet

include("../scripts/tripletloss/muta-train.jl")
export plot_process
export params_importance

# --------------------------------------------------------------

include("../scripts/clustering/em-script.jl")
export test_em
export main_em_gaussian
export main_em_exponential
export main_em_uniform
export main_em_laplace

include("../scripts/clustering/lac-script.jl")
export test_h
export test_lac
export main_lac_gaussian
export main_lac_exponential
export main_lac_uniform
export main_lac_laplace

# --------------------------------------------------------------

include("../scripts/lat-script.jl")
export test_lat

# --------------------------------------------------------------

include("utils/dataloading.jl")
export load

include("utils/datagen.jl")
export generate_data_2d
export generate_separable_2d

include("utils/plotfuncs.jl")
export plot_data
export plot_process
export plot_classes_2d
export plot_distributions_2d

# --------------------------------------------------------------

function test()
    Pkg.test("HierarchicalMetric.jl")
end
export test

end  # HierarchicalMetric