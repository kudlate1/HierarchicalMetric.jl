ChamferDistance(pm::AbstractArray{<:Real}) = mean(minimum(pm, dims=1)) + mean(minimum(pm, dims=2))

function HausdorffDistance(c)
    d₁ = maximum(minimum(c, dims=1))
    d₂ = maximum(minimum(c, dims=2))
    maximum([d₁, d₂])
end

function Wasserstein(c::AbstractMatrix, px, py; optimizer=Tulip.Optimizer())
    return OptimalTransport.emd2(px, py, c, optimizer)
end 

function WassersteinProbDist(c::AbstractMatrix; optimizer=Tulip.Optimizer())
    ls₁, ls₂ = size(c)
    p = fill(1/ls₁, ls₁);
    q = fill(1/ls₂, ls₂);
    Wasserstein(c, p, q; optimizer=optimizer)
end 


"""
    WassersteinMultiset(c::AbstractMatrix; optimizer=Tulip.Optimizer())

Function computes Wasserstein Distance with while cardinality metters!!

We assume that input \"c\" already include \"norms\" as last row if size(c,1) < size(c, 2)
or last column if size(c,1) > size(c, 2).

Theoretical Example 1: \\
    𝒞 ≈ 2x4 Matrix \\
    then 2nd row is expected to be \"norms\" \\
    then 𝓅 = [0.25, 0.75] and 𝓆 = [0.25, 0.25, 0.25, 0.25]
    and vice versa
    
Theoretical Example 2: \\
    𝒞 ≈ 4x3 Matrix \\
    then 3rd column is expected to be \"norms\" \\
    then 𝓅 = [0.25, 0.25, 0.25, 0.25] and 𝓆 = [0.25, 0.25, 0.5]
    and vice versa

OptimalTransport.emd2(𝓅, 𝓆, 𝒞)
"""
function WassersteinMultiset(c::AbstractMatrix; optimizer=Tulip.Optimizer())
    ls₁, ls₂ = size(c);
    if ls₁ < ls₂
        p = vcat(ones(Float64, ls₁-1), [ls₂-(ls₁-1)]) ./ ls₂; # everything has to be in Float64!!! Float32 cause ERROR
        q = fill(1/ls₂, ls₂);
    elseif ls₁ > ls₂
        p = fill(1/ls₁, ls₁);
        q = vcat(ones(Float64, ls₂-1), [ls₁-(ls₂-1)]) ./ ls₁;
    else
        p = fill(1/ls₁, ls₁);
        q = fill(1/ls₂, ls₂);
    end
    Wasserstein(c, p, q; optimizer=optimizer)
end 
