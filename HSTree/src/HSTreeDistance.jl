module HSTreeDistance

using Statistics, Distances, StringDistances
using Flux, Zygote, Mill, OneHotArrays, MLUtils
using ChainRulesCore, HierarchicalUtils
using OptimalTransport, Tulip, LinearAlgebra


include("utils.jl")
export zerocardinality, preprocess_empty_bags, preprocess_missing, pad_leaves_for_wasserstein

include("building_blocks/leaf_metrics.jl")
export Pairwise_Euclidean, Pairwise_SqEuclidean
export Pairwise_Cityblock
export Pairwise_Levenstein, NormLevenshtein 

include("building_blocks/product_metrics.jl")
export WeightedProductMetric, SqWeightedProductMetric

include("building_blocks/bag_metrics.jl")
export ChamferDistance, HausdorffDistance, WassersteinProbDist, WassersteinMultiset

include("building_blocks/cardinality_metrics.jl")
export ScaleOne, MaxCard

include("weights_struct.jl")
export WeightStruct, destructure_metric_to_ws

include("block_segmented_norm/chamfer_distance.jl")
include("block_segmented_norm/hausdorff_distance.jl")
include("block_segmented_norm/wasserstein_distance.jl")
export WassersteinCoeffs

include("block_segmented_norm/block_segmented_norms.jl")
export block_segmented_norm

include("metric.jl")
export AbstractMetric, LeafMetric, ProductMetric, SetMetric

include("reflectmetric.jl")
export reflectmetric, _reflectmetric

include("printing.jl")


end # module HSTreeDistance
