using Test

function testClassSplit()

    @testset "Testing splitClasses function" verbose=true begin
        
        @testset "Testing basic functionality" begin
        
            product_nodes = ["node1", "node2", "node3", "node4"]
            y = ["A", "B", "A", "A"]
            anchor = "node1"
            anchor_label = "A"

            positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

            expected_positives = ["node3", "node4"]
            expected_negatives = ["node2"]

            @test positives == expected_positives
            @test negatives == expected_negatives
        end
        
        @testset "Testing edge case: only one node in a batch" begin
            
            product_nodes = ["node1"]
            y = ["A"]
            anchor = "node1"
            anchor_label = "A"

            positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

            expected_positives = []
            expected_negatives = []

            @test positives == expected_positives
            @test negatives == expected_negatives
        end

        @testset "Testing edge case: no positives nodes in a batch" begin 

            product_nodes = ["node1", "node2", "node3", "node4"]
            y = ["A", "B", "A", "A"]
            anchor = "node2"
            anchor_label = "B"
            positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)

            expected_positives = []
            expected_negatives = ["node1", "node3", "node4"]

            @test positives == expected_positives
            @test negatives == expected_negatives
        end

        @testset "Testing edge case: no negatives nodes in a batch" begin 

            product_nodes = ["node1", "node2", "node3", "node4"]
            y = ["A", "A", "A", "A"]
            anchor = "node2"
            anchor_label = "A"
            positives, negatives = separateClasses(product_nodes, y, anchor, anchor_label)
    
            expected_positives = ["node1", "node3", "node4"]
            expected_negatives = []
    
            @test positives == expected_positives
            @test negatives == expected_negatives
        end
    end
end

testClassSplit();

