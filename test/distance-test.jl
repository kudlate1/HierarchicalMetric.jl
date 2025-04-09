using Test

function test_distance()

    X = Float64.([1 -3 0 0 1 2 4 6 0 0; 2 4 0 1 1 3 -3 5 0 2])
    PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
    product_nodes = [PN[i] for i in 1:10]
    PM = reflectmetric(product_nodes[1])

    @testset "Testing correct distance computation" verbose=true begin
        
        @testset "Testing basic case" begin
            @test distance(PN[3], PN[4], PM) == 1.0
            @test distance(PN[3], PN[10], PM) == 2.0
        end

        @testset "Testing distance of the two same nodes" begin
            @test -1e-9 < distance(PN[1], PN[1], PM) < 1e-9
            @test -1e-9 < distance(PN[2], PN[2], PM) < 1e-9
            @test -1e-9 < distance(PN[9], PN[9], PM) < 1e-9
        end

        @testset "Testing precision" begin
            @test distance(PN[3], PN[5], PM) ≈ sqrt(2.0)
        end

    end

    @testset "Testing pairwise matrix" verbose=true begin
        
        @testset "Testing correct output type of pairwiseDistance function" begin
            matrix1 = pairwise_distance(product_nodes)
            matrix2 = pairwise_distance([product_nodes[1]])
            matrix3 = pairwise_distance([])
            @test matrix1 |> typeof == Matrix{Float64}
            @test matrix2 |> typeof == Matrix{Float64}
            @test matrix3 |> typeof == Matrix{Float64}
        end

        @testset "Testing correct computation of matrix" begin
            X = Float64.([0 0 0; 1 2 3])
            PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
            product_nodes = [PN[i] for i in 1:3]
            @test round.(pairwise_distance(product_nodes; wt=identity), digits=5) ≈ [0.0 1.0 2.0;
                                                                                    1.0 0.0 1.0;
                                                                                    2.0 1.0 0.0]
            X = Float64.([0 0 0; 1 2 3])
            PN = ProductNode((x = Array(X[1, :]'), y  = Array(X[2, :]')))
            product_nodes = [PN[i] for i in 1:3]
            @test round.(pairwise_distance(product_nodes; wt=softplus), digits=5) ≈ [0.0      1.14598  2.29195;
                                                                                    1.14598  0.0      1.14598;
                                                                                    2.29195  1.14598  0.0]                                              
        end
    end
end
