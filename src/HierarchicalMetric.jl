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
export pairwiseDistance
export selectTriplet
export tripletLoss
export distance
export splitClasses

include("datatraining.jl")
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

function test()
    Pkg.test("HierarchicalMetric.jl")
end

export test

include("../scripts/triplet-triv.jl")
export plotData
export createProductNodes
export visualiseDistances
export plotProcess

include("../scripts/muta-train.jl")
export plotProcess
export paramsImportance

include("../scripts/clustering.jl")
export plot_classes_2d
export metrics
export weight_transform
export plot_distributions_2d
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