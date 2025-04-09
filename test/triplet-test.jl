using Test

function test_triplet_selection()

    X = Float64.([1 -3 0 0 1 2 4 6 0 0; 2 4 0 1 1 3 -3 5 0 2])
    PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
    product_nodes = [PN[i] for i in 1:10]
    PM = reflectmetric(product_nodes[1])
    dists = pairwise_Distance(product_nodes)

    @testset "Testing selectTriplet function" verbose=true begin

        @testset "Testing basic case of return" begin
            y = [1 1 1 1 1 0 0 0 0 0]
            @test (select_triplet(SelectRandom(), dists, product_nodes, y, PM)) |> typeof <: Tuple{ProductNode, ProductNode, ProductNode}
            @test (select_triplet(SelectHard(), dists, product_nodes, y, PM)) |> typeof <: Tuple{ProductNode, ProductNode, ProductNode}
        end

        @testset "Testing edge case: negative not found" begin
            # possible to choose a batch of just one class -> triplet is incomplete, should be treated in other funcs
            y = [1 1 1 1 1 1 1 1 1 1]
            @test (select_triplet(SelectRandom(), dists, product_nodes, y, PM)) |> typeof <: Tuple{ProductNode, ProductNode, Bool}
            @test (select_triplet(SelectHard(), dists, product_nodes, y, PM)) |> typeof <: Tuple{ProductNode, ProductNode, Bool}
        end

        @testset "Testing edge case: positive not found" begin
            # both SelectHard() and SelectRandom() repeats anchor selecting -> returns always complete triplet 
            y = [0 1 1] # if anchor = 0 -> positive class is empty
            @test (select_triplet(SelectRandom(), dists, product_nodes[1:3], y, PM)) |> typeof <: Tuple{ProductNode, ProductNode, ProductNode}
            @test (select_triplet(SelectHard(), dists, product_nodes[1:3], y, PM)) |> typeof <: Tuple{ProductNode, ProductNode, ProductNode}
        end
    end
end
