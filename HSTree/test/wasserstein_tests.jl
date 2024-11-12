using Pkg, Revise#, DrWatson
#@quickactivate
using BenchmarkTools, Test, Random
using Mill, HSTreeDistance, Distances
using OptimalTransport, Flux, Zygote


# define Common data for whole test file
Random.seed!(666)
a1_, b1_, b2_ = randn(3,1), randn(2,4), randn(4,6);
a11_, b11_, b21_ = randn(3,1), randn(2,3), randn(4,5);

t1 = ProductNode((
    a1 = ArrayNode(a1_),
    a2 = BagNode(
        ProductNode(
            b1=ArrayNode(b1_), 
            b2=BagNode(
                ArrayNode(b2_), 
                AlignedBags([1:3, 4:4, 0:-1, 5:6]))), 
        AlignedBags(1:4))
    ))

t2 = ProductNode((
    a1 = ArrayNode(a11_),
    a2 = BagNode(
        ProductNode(
            b1=ArrayNode(b11_), 
            b2=BagNode(
                ArrayNode(b21_), 
                AlignedBags([1:1, 0:-1, 2:5]))), 
        AlignedBags(1:3))
))


# define bag and metric for bag
b = pad_leaves_for_wasserstein(BagNode(ArrayNode(b2_), AlignedBags([1:3, 4:4, 0:-1, 5:6])))
b̃ = pad_leaves_for_wasserstein(BagNode(ArrayNode(b21_), AlignedBags([1:1, 0:-1, 2:5])))

@testset "Forward Pass and Correctness of Wasserstein Distance" verbose=true begin

    # leaves
    @testset "Initial test for Leaf Metric" begin 
        mb = reflectmetric(b, set_metric=WassersteinMultiset)
        c₁ = pairwise(Euclidean(), hcat(b2_, zeros(size(b2_, 1), 1)), hcat(b21_, zeros(size(b21_, 1), 1)))
        c̃₁ = mb.im(b.data, b̃.data)
        @test c₁ ≈ c̃₁
    end
    # bag
    @testset "WassersteinMultiset (ScaleOne) on simple BagNode" begin # TODO revise me
        mb = reflectmetric(b, set_metric=WassersteinMultiset)
        WM = WassersteinMultiset;
        c₁ = pairwise(Euclidean(), hcat(b2_, zeros(size(b2_, 1), 1)), hcat(b21_, zeros(size(b21_, 1), 1)))

        ĉ₂ = mb(b, b̃)
        c̃₂ = block_segmented_norm(mb.im(b.data, b̃.data), b.bags, b̃.bags, WassersteinMultiset)
        c₂ = [ 
            WM(c₁[1:3,[1,6]])    WM(c₁[1:3,6:6])    WM(c₁[[1,2,3,7], 2:5]);#   WM(c₁[1:3,6:6]);
            WM(c₁[4:4,1:1])      WM(c₁[4:4,6:6])    WM(c₁[[4,7], 2:5]);    #   WM(c₁[4:4,6:6]);
            WM(c₁[7:7, 1:1])     WM(c₁[7:7,6:6])    WM(c₁[7:7,2:5]);       #   WM(c₁[7:7,6:6]);
            WM(c₁[5:6, [1,6]])   WM(c₁[5:6,6:6])    WM(c₁[[5,6,7], 2:5]);  #   WM(c₁[5:6,6:6]);
            #WM(c₁[7:7,1:1])      WM(c₁[7:7,6:6])    WM(c₁[7:7,2:5])          WM(c₁[7:7,6:6]); 
            ]

        @test c̃₂ ≈ ĉ₂
        @test c₂ ≈ c̃₂
        @test c₂ ≈ ĉ₂
    end
    @testset "WassersteinProbDist (ScaleOne) on simple BagNode" begin 
        mb = reflectmetric(b, set_metric=WassersteinProbDist)
        c₁ = pairwise(Euclidean(), hcat(b2_, zeros(size(b2_, 1), 1)), hcat(b21_, zeros(size(b21_, 1), 1)))
        c̃₃ = block_segmented_norm(mb.im(b.data, b̃.data), b.bags, b̃.bags, WassersteinProbDist)
        c₃ = [WassersteinProbDist(c₁[b1, b2]) for b1 in b.bags, b2 in b̃.bags]
        ĉ₃ = mb(b, b̃)

        @test c̃₃ ≈ c₃
        @test ĉ₃ ≈ c₃
        @test ĉ₃ ≈ c̃₃
    end

    @testset "WassersteinMultiset (MaxCard) on simple BagNode" begin
        mb = reflectmetric(b, set_metric=WassersteinMultiset, card_metric=MaxCard)
        WM = WassersteinMultiset;
        c₁ = pairwise(Euclidean(), hcat(b2_, zeros(size(b2_, 1), 1)), hcat(b21_, zeros(size(b21_, 1), 1)))

        ĉ₄ = mb(b, b̃)
        M, bias = MaxCard(b.bags, b̃.bags)
        c̃₄ = M .* block_segmented_norm(mb.im(b.data, b̃.data), b.bags, b̃.bags, WassersteinMultiset) .+ bias
        c₄ = [ 
            3 * WM(c₁[1:3,[1,6]])    3 * WM(c₁[1:3,6:6])    4 * WM(c₁[[1,2,3,7], 2:5]);#   3 * WM(c₁[1:3, 6:6]);
            1 * WM(c₁[4:4,1:1])      1 * WM(c₁[4:4,6:6])    4 * WM(c₁[[4,7], 2:5])    ;#   1 * WM(c₁[4:4,6:6]);
            1 * WM(c₁[7:7, 1:1])     1 * WM(c₁[7:7,6:6])    4 * WM(c₁[7:7,2:5])       ;#   1 * WM(c₁[7:7,6:6]);
            2 * WM(c₁[5:6, [1,6]])   2 * WM(c₁[5:6,6:6])    4 * WM(c₁[[5,6,7], 2:5])  ;#   2 * WM(c₁[5:6,6:6]);
            #1 * WM(c₁[7:7,1:1])      1 * WM(c₁[7:7,6:6])    4 * WM(c₁[7:7,2:5])           1 * WM(c₁[7:7,6:6]); 
            ]

        @test c̃₄ ≈ ĉ₄
        @test c₄ ≈ c̃₄
        @test c₄ ≈ ĉ₄
    end

    @testset "WassersteinProbDist (MaxCard) on simple BagNode" begin 
        mb = reflectmetric(b, set_metric=WassersteinProbDist, card_metric=MaxCard)
        c₁ = pairwise(Euclidean(), hcat(b2_, zeros(size(b2_, 1), 1)), hcat(b21_, zeros(size(b21_, 1), 1)))
        M, bias = MaxCard(b.bags, b̃.bags)
        c̃₃ = M .* block_segmented_norm(mb.im(b.data, b̃.data), b.bags, b̃.bags, WassersteinProbDist) .+ bias
        c₃ = [maximum([length(b1), length(b2)]) .* WassersteinProbDist(c₁[b1, b2]) for b1 in b.bags, b2 in b̃.bags]
        ĉ₃ = mb(b, b̃)

        @test c̃₃ ≈ c₃
        @test ĉ₃ ≈ c₃
        @test ĉ₃ ≈ c̃₃
    end
end


@testset "Gradients and Backward Pass of Wasserstein Distances" verbose=true begin  # TODO revise me
    t₁ = pad_leaves_for_wasserstein(t1)
    t₂ = pad_leaves_for_wasserstein(t2) 
    # Helper functions
    grad_norm(x::Vector) = sqrt(sum(abs2.(x)))
    grad_norm(g::Tuple) = grad_norm(Flux.destructure(g)[1]) # Flux.destructure(g)[1]; # separate values from gradient struct
    grad_norm(g::Zygote.Grads, ps::Flux.Params) = grad_norm(vcat([g.grads[ps[i]] for i ∈ 1:length(ps)]...))
    ps2vec(ps::Flux.Params) = vcat([ps[i] for i ∈ 1:length(ps)]...)
    grad2vec(g::Zygote.Grads, ps::Flux.Params) = vcat([g.grads[ps[i]] for i ∈ 1:length(ps)]...)

    # Simple Bags
    @testset "Simple BagNode" verbose=true begin
        @testset "WassersteinMultiset & ScaleOne" begin
            mb = reflectmetric(b, set_metric=WassersteinMultiset, card_metric=ScaleOne)
            g = gradient((b)->sum(mb(b, b̃)), b);
            @test g !== nothing
            @test grad_norm(g) ≈ 2.9636279412101234#4.616867196618156
        end
        @testset "WassersteinProbDist & ScaleOne" begin 
            mb = reflectmetric(b, set_metric=WassersteinProbDist, card_metric=ScaleOne)
            g = gradient((b)->sum(mb(b, b̃)), b);
            @test g !== nothing
            @test grad_norm(g) ≈ 3.1793709021606236#4.816843581322361
        end
        @testset "WassersteinMultiset & MaxCard" begin
            mb = reflectmetric(b, set_metric=WassersteinMultiset, card_metric=MaxCard)
            g = gradient((b)->sum(mb(b, b̃)), b);
            @test g !== nothing
            @test grad_norm(g) ≈ 7.363662612357174#10.492053137192054
        end 
        @testset "WassersteinProbDist & MaxCard" begin 
            mb = reflectmetric(b, set_metric=WassersteinProbDist, card_metric=MaxCard)
            g = gradient((b)->sum(mb(b, b̃)), b);
            @test g !== nothing
            @test grad_norm(g) ≈ 7.583629524437483#10.53815026354742
        end
    end

    # Complex data
    @testset "Complex data" verbose=true begin
        @testset "WassersteinMultiset & ScaleOne" begin
            mt = reflectmetric(t1, set_metric=WassersteinMultiset, card_metric=ScaleOne)
            ps = Flux.params(mt);
            g = gradient(()->sum(mt(t₁, t₂)), ps)
            @test g !== nothing
            @test grad_norm(g, ps) ≈ 1.0124645f0
            Flux.update!(ADAM(), ps, g); # Update
            @test all(ps2vec(ps) .!= 1f0)
        end
        @testset "WassersteinProbDist & ScaleOne" begin
            mt = reflectmetric(t1, set_metric=WassersteinProbDist, card_metric=ScaleOne)
            ps = Flux.params(mt);
            g = gradient(()->sum(mt(t₁, t₂)), ps)
            @test g !== nothing
            @test grad_norm(g, ps) ≈ 1.211407f0
            Flux.update!(ADAM(), ps, g); # Update
            @test all(ps2vec(ps) .!= 1f0)
            
        end
        @testset "WassersteinMultiset & MaxCard" begin
            mt = reflectmetric(t1, set_metric=WassersteinMultiset, card_metric=MaxCard)
            ps = Flux.params(mt);
            g = gradient(()->sum(mt(t₁, t₂)), ps)
            @test g !== nothing
            @test grad_norm(g, ps) ≈ 10.0105715f0
            Flux.update!(ADAM(), ps, g); # Update
            @test all(ps2vec(ps) .!= 1f0)
        end
        @testset "WassersteinProbDist & MaxCard" begin
            mt = reflectmetric(t1, set_metric=WassersteinProbDist, card_metric=MaxCard)
            ps = Flux.params(mt);
            g = gradient(()->sum(mt(t₁, t₂)), ps)
            @test g !== nothing
            @test grad_norm(g, ps) ≈ 12.700179f0
            Flux.update!(ADAM(), ps, g); # Update
            @test all(ps2vec(ps) .!= 1f0)
        end
    end
end