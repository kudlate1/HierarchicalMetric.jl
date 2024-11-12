# Block Segmened Norm with WassersteinDistance and WassersteinMultiset
function forward_block_segmented_wass(x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags, multiset::Bool)#; optimizer=Tulip.Optimizer()
    dimx, dimy = size(x);
    ls1 ,ls2 = length(seg₁), length(seg₂)
    o = similar(x, (ls1, ls2));
    Γ = zeros_like(x)
    segments = Array{Tuple}(undef, ls1*ls2)
    for (i,sᵢ) in enumerate(seg₁)
        for (j,sⱼ) in enumerate(seg₂)
            γ, sᵢₙ, sⱼₙ = WassersteinCoeffs(x, sᵢ, sⱼ, dimx, dimy, multiset)
            o[i,j] = LinearAlgebra.dot(γ, x[sᵢₙ, sⱼₙ]) 
            if sᵢ != sᵢₙ  && unique(sᵢₙ) == [dimx]
                Γ[sᵢ, sⱼₙ] .+= sum(γ, dims=1)
                segments[(i-1)*ls2 + j] = (i, j, sᵢ, sⱼₙ);#push!(segments, (i, j, sᵢ, sⱼₙ))
            elseif sⱼ != sⱼₙ && unique(sⱼₙ) == [dimy]
                Γ[sᵢₙ, sⱼ] .+= sum(γ, dims=2)
                segments[(i-1)*ls2 + j] = (i, j, sᵢₙ, sⱼ)#push!(segments, (i, j, sᵢₙ, sⱼ))
            else
                Γ[sᵢₙ,sⱼₙ] .+= γ
                segments[(i-1)*ls2 + j] =(i, j, sᵢₙ, sⱼₙ)#push!(segments, (i, j, sᵢₙ, sⱼₙ))
            end
        end
    end
    return o, Γ, segments
end

function backward_block_segmented_wass(Δ, x, Γ, segments)
    o = zeros_like(x)
    for (i, j, sᵢ, sⱼ) in segments
        o[sᵢ, sⱼ] .= Δ[i,j]
        # Justification of usage of ".=" instead of ".+=" (with normalization later): 
        # By our design all subseqments are unique except or "norms" last column and last row. 
        # Therefore (N-1)x(M-1) matrix is unique without overwritting. 
        # Last row and last column doesn't need to be sumed here, because their gradients (γ-s) were sumed already in forwad pass
    end
    # input to rrule has 5 arguments -> therefore 4 times NoTangent()
    return NoTangent(), Γ .* o, NoTangent(), NoTangent(), NoTangent()
end


# Helper Functions

function WassersteinCoeffs(x::AbstractMatrix, seg₁, seg₂, dimx, dimy, multiset::Bool=false; optimizer=Tulip.Optimizer())
    if multiset
        WassersteinCoeffsMultiset(x, seg₁, seg₂, dimx, dimy; optimizer=optimizer)
    else
        WassersteinCoeffsProbDist(x, seg₁, seg₂; optimizer=optimizer) # , dimx, dimy;
    end
end

function WassersteinCoeffsProbDist(x::AbstractMatrix, seg₁, seg₂; optimizer=Tulip.Optimizer())
    ls₁, ls₂ = length(seg₁), length(seg₂)
    p = fill(1/ls₁, ls₁);
    q = fill(1/ls₂, ls₂);
    c = Float64.(x[seg₁, seg₂]); # everything has to be in Float64!!! Float32 cause ERROR
    γ = OptimalTransport.emd(p, q, c, optimizer)
    return γ, seg₁, seg₂
end

function WassersteinCoeffsMultiset(x::AbstractMatrix, seg₁, seg₂, dimx, dimy; optimizer=Tulip.Optimizer())
    ls₁, ls₂ = length(seg₁), length(seg₂)
    if ls₁ < ls₂
        #@info "ls₁ < ls₂"
        seg₁ₙ = vcat(seg₁, [dimx]);
        seg₂ₙ = seg₂;
        p = vcat(ones(Float64, ls₁), [ls₂-ls₁]) ./ ls₂; # everything has to be in Float64!!! Float32 cause ERROR
        q = fill(1/ls₂, ls₂);
    elseif ls₁ > ls₂
        #@info "ls₁ > ls₂"
        seg₁ₙ = seg₁;
        seg₂ₙ = vcat(seg₂, [dimy]);
        p = fill(1/ls₁, ls₁);
        q = vcat(ones(Float64, ls₂), [ls₁-ls₂]) ./ ls₁; # everything has to be in Float64!!! Float32 cause ERROR
    else
        #@info "ls₁ == ls₂"
        seg₁ₙ = seg₁;
        seg₂ₙ = seg₂;
        p = fill(1/ls₁, ls₁);
        q = fill(1/ls₂, ls₂);
    end
    c = Float64.(x[seg₁ₙ, seg₂ₙ]); # everything has to be in Float64!!! Float32 cause ERROR
    γ = OptimalTransport.emd(p, q, c, optimizer)
    return γ, seg₁ₙ, seg₂ₙ
end


function forward_block_segmented_wass_quick(x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags, multiset::Bool)
    dimx, dimy = size(x);

    function wasserstein(x, sᵢ, sⱼ)
        γ, sᵢₙ, sⱼₙ = WassersteinCoeffs(x, sᵢ, sⱼ, dimx, dimy, multiset)
        return LinearAlgebra.dot(γ, x[sᵢₙ, sⱼₙ])
    end
    
    [wasserstein(x, sᵢ, sⱼ) for sᵢ in seg₁, sⱼ in seg₂]
end

