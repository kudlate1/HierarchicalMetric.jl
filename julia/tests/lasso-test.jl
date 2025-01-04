using Test

# test if there any gradients from lasso objective
function test_lasso(method::SelectingTripletMethod; λ = 0.01)

    X, y = load("julia/data/mutagenesis.json")
    metric = reflectmetric(X[1], weight_sampler=randn, weight_transform=softplus) 
    anchor, pos, neg = selectTriplet(method, distances, X[1:10], y[1:10], metric) # just on small batch
        
    @testset "Testing gradients of triplet loss with lasso" verbose=true begin
        @testset "testing contribution of only lasso λₗₐₛₛₒ = 1.0" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            ps = Flux.params(metric)
            opt = Descent(λ)
            loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; α=-Inf, λₗₐₛₛₒ=1f0, weight_transform=identity), ps)
            # if α= -Inf or number small enaugh, the triplet loss contribution to loss function will be zero
            # And loss will be equal to regularization term from lasso


            @test loss ≈ sqrt(sum(ones(13))) # sqrt(13)
            @test reduce(vcat, ps) == ones(13)
            Flux.update!(opt, ps, grad) # check if parameters after output are those what we wished for
            @test reduce(vcat, ps) ≈ ones(13) .- λ * sqrt(13)/13
        
        end

        @testset "testing contribution of lasso λₗₐₛₛₒ = 0.5" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            ps = Flux.params(metric)
            opt = Descent(λ)
            loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; α=-Inf, λₗₐₛₛₒ=0.5, weight_transform=identity), ps)

            @test loss ≈ 0.5*sqrt(sum(ones(13))) # sqrt(13)
            @test reduce(vcat, ps) == ones(13)
            Flux.update!(opt, ps, grad) # check if parameters after output are those what we wished for
            @test reduce(vcat, ps) ≈ ones(13) .- λ * sqrt(13)/13*0.5
        
        end
        
        @testset "testing contribution of tripletloss together with lasso λₗₐₛₛₒ = 1.0" begin
            metric = reflectmetric(X[1], weight_sampler=ones, weight_transform=identity)
            ps = Flux.params(metric)
            opt = Descent(λ)
            loss, grad = Flux.withgradient(() -> tripletLoss(anchor, pos, neg, metric; α=1, λₗₐₛₛₒ=1f0, weight_transform=identity), ps)

            @test !(loss ≈ 1f0 * sqrt(sum(ones(13)))) # sqrt(13)
            @test reduce(vcat, ps) == ones(13)
            Flux.update!(opt, ps, grad) # check if parameters after output are those what we wished for
            @test !(reduce(vcat, ps) ≈ ones(13) .- λ * sqrt(13)/13)
        
        end
    end
end

test_lasso(SelectHard());