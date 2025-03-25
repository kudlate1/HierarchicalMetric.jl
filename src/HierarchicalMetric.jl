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

include("../scripts/LAC-triv.jl")
export plot_points

include("../scripts/clustering.jl")
export generate_dataset
export plot_classes
export metrics
export weight_transform

end  # HierarchicalMetric