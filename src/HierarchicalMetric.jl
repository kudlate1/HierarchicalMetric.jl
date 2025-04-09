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


# structure for triplet selection
abstract type TripletSelectionMethod end
struct SelectRandom <: TripletSelectionMethod end
struct SelectHard <: TripletSelectionMethod end
export TripletSelectionMethod, SelectHard, SelectRandom

include("triplet-loss.jl")
export pairwise_distance
export select_triplet
export triplet_loss
export distance
export split_classes

include("triplettrain.jl")
export train

include("dataloading.jl")
export load

include("lac.jl")
export LAC
export kmeanspp

include("em.jl")
export EM_GMM

include("datagen.jl")
export generate_dataset_2d
export generate_exponential_2d
export generate_uniform_2d
export generate_laplace_2d

include("plotfuncs.jl")
export plot_data
export plot_process
export plot_classes_2d
export plot_distributions_2d

function test()
    Pkg.test("HierarchicalMetric.jl")
end

export test

include("../scripts/triplet-triv.jl")
export create_product_nodes
export visualise_distances

include("../scripts/muta-train.jl")
export plot_process
export params_importance

include("../scripts/clustering.jl")
export metrics
export weight_transform
export means_precision
export covariances_precision
export main_lac_gaussian
export main_lac_exponential
export main_lac_uniform
export main_lac_laplace
export main_em_gaussian
export main_em_exponential
export main_em_uniform
export main_em_laplace


end  # HierarchicalMetric