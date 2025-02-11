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

    testClassSplit()

    testLasso(SelectRandom())

    testTripletSelection()

    testDistance()

end