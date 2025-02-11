using Test

# test if there any gradients from lasso objective
function testLasso(method::TripletSelectionMethod; λ = 0.01)

    X, y = load("../data/mutagenesis.json")
    metric = reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) 
    distances = pairwiseDistance(X[1:10])
    anchor, pos, neg = selectTriplet(method, distances, X[1:10], y[1:10], metric) # just on small batch
        
    @testset "Testing gradients of triplet loss with lasso" verbose=true begin
        
        @testset "testing contribution of only lasso λₗₐₛₛₒ = 1.0" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            opt = Descent(λ)
            state_tree = Flux.setup(opt, metric)
            loss, grad = Flux.withgradient(metric) do m
                tripletLoss(anchor, pos, neg, m; α=-Inf, λₗₐₛₛₒ=1f0, weight_transform=identity)
            end
            # if α= -Inf or number small enough, the triplet loss contribution to loss function will be zero
            # And loss will be equal to regularization term from lasso

            @test loss ≈ sum(ones(13)) # sqrt(13)
            @test Flux.destructure(metric)[1] == ones(13)
            Flux.update!(state_tree, metric, grad[1]) # check if parameters after output are those what we wished for
            @test Flux.destructure(metric)[1] ≈ ones(13) .- λ * 13/13
        
        end

        @testset "testing contribution of lasso λₗₐₛₛₒ = 0.5" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            opt = Descent(λ)
            state_tree = Flux.setup(opt, metric)
            loss, grad = Flux.withgradient(metric) do m
                tripletLoss(anchor, pos, neg, m; α=-Inf, λₗₐₛₛₒ=0.5, weight_transform=identity)
            end
            
            @test loss ≈ 0.5*sum(ones(13)) # sqrt(13)
            @test Flux.destructure(metric)[1]  == ones(13)
            Flux.update!(state_tree, metric, grad[1]) # check if parameters after output are those what we wished for
            @test Flux.destructure(metric)[1]  ≈ ones(13) .- λ * 13/13*0.5
        
        end
        
        @testset "testing contribution of tripletloss together with lasso λₗₐₛₛₒ = 1.0" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            opt = Descent(λ)
            state_tree = Flux.setup(opt, metric)
            loss, grad = Flux.withgradient(metric) do m
                tripletLoss(anchor, pos, neg, m; α=-Inf, λₗₐₛₛₒ=1f0, weight_transform=identity)
            end

            @test loss ≈ 1f0 * sum(ones(13)) # sqrt(13)
            @test Flux.destructure(metric)[1]  == ones(13)
            Flux.update!(state_tree, metric, grad[1]) # check if parameters after output are those what we wished for
            @test Flux.destructure(metric)[1]  ≈ ones(13) .- λ * 13/13*1f0
        
        end
    end
end
