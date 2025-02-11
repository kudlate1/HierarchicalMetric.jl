
function forward_block_segmented_cd(x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags) 
    o = similar(x, (length(seg₁), length(seg₂)))
    argmins₁ = Matrix{CartesianIndex{2}}(undef, length(seg₁), size(x, 2))
    argmins₂ = Matrix{CartesianIndex{2}}(undef, size(x, 1), length(seg₂))
    for (i,sᵢ) in enumerate(seg₁)
        for (j,sⱼ) in enumerate(seg₂)
            min₁, argmin₁ = findmin(x[sᵢ, sⱼ], dims=1)
            min₂, argmin₂ = findmin(x[sᵢ, sⱼ], dims=2)
            o[i,j] = mean(min₁) + mean(min₂)
            argmins₁[i, sⱼ] .= argmin₁[:]
            argmins₂[sᵢ, j] .= argmin₂[:]
        end
    end
    return o, argmins₁, argmins₂
end

function backward_block_segmented_cd(ȳ, x, seg₁, seg₂, argmins₁, argmins₂)
    o = zero(x)
    for (i, sᵢ) in enumerate(seg₁)
        for (j, sⱼ) in enumerate(seg₂)
            segment = o[sᵢ, sⱼ] 
            segment[argmins₂[sᵢ, j]] .+= ȳ[i, j] / length(sᵢ)
            segment[argmins₁[i, sⱼ]] .+= ȳ[i, j] / length(sⱼ)
            o[sᵢ, sⱼ] .= segment # updated segment 
        end
    end
    return NoTangent(), o, NoTangent(), NoTangent(), NoTangent()
end