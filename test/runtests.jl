using Test

using HierarchicalMetric
using HierarchicalMetric.Flux
using HierarchicalMetric.HSTreeDistance
using HierarchicalMetric.Mill

include("lasso-test.jl")
include("class-split-test.jl")
include("distance-test.jl")
include("triplet-test.jl")


@testset "HierarchicalMetric.jl - running all the package tests" begin

    test_class_split()

    test_lasso(SelectRandom())

    test_lasso(SelectHard())

    test_triplet_selection()
    
    test_distance()

end