# Define shortcuts
WPD = typeof(WassersteinProbDist)
WMS = typeof(WassersteinMultiset)
WassFamily = Union{WPD, WMS}


# General Formula
function block_segmented_norm(x, seg₁, seg₂, norm::Function=sum)
    [norm(x[sᵢ, sⱼ]) for sᵢ in seg₁, sⱼ in seg₂]
end


# ChamferDistance
function block_segmented_norm(x::AbstractMatrix{<:Real}, seg₁, seg₂, ::typeof(ChamferDistance))
    forward_block_segmented_cd(x, seg₁, seg₂)[1]
end

function ChainRulesCore.rrule(::typeof(block_segmented_norm), x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags, ::typeof(ChamferDistance))
    y, argmins₁, argmins₂ = forward_block_segmented_cd(x, seg₁, seg₂)
    grad = Δ -> backward_block_segmented_cd(Δ, x, seg₁, seg₂, argmins₁, argmins₂)
    return y, grad
end

# Hausdorff Distance
function block_segmented_norm(x::AbstractMatrix{<:Real}, seg₁, seg₂, ::typeof(HausdorffDistance))
    forward_block_segmented_hausdorff(x, seg₁, seg₂)[1]
end

function ChainRulesCore.rrule(::typeof(block_segmented_norm), x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags, ::typeof(HausdorffDistance))
    y, argmaxmins = forward_block_segmented_hausdorff(x, seg₁, seg₂)
    grad = Δ -> backward_block_segmented_hausdorff(Δ, x, seg₁, seg₂, argmaxmins)
    return y, grad
end

# Wasserstein Distance
function block_segmented_norm(x::AbstractMatrix{<:Real}, seg₁, seg₂, f::Union{WPD, WMS})
    forward_block_segmented_wass(x, seg₁, seg₂, f isa WMS)[1]
end

function ChainRulesCore.rrule(::typeof(block_segmented_norm), x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags, f::Union{WPD, WMS})
    y, Γ, segments = forward_block_segmented_wass(x, seg₁, seg₂, f isa WMS)
    grad = Δ -> backward_block_segmented_wass(Δ, x, Γ, segments)
    return y, grad
end