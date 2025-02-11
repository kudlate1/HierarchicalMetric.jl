function forward_block_segmented_hausdorff(x::AbstractMatrix, seg₁::AbstractBags, seg₂::AbstractBags) 
    o = similar(x, (length(seg₁), length(seg₂)))
    indexes = Matrix{CartesianIndex{2}}(undef, length(seg₁), length(seg₂))
    for (i, sᵢ) in enumerate(seg₁)
        for (j, sⱼ) in enumerate(seg₂)
            min₁, argmin₁ = findmin(x[sᵢ, sⱼ], dims=1)
            min₂, argmin₂ = findmin(x[sᵢ, sⱼ], dims=2)
            max₁, argmax₁ = findmax(min₁)
            max₂, argmax₂ = findmax(min₂)
            argmaxmins = [argmin₁[argmax₁], argmin₂[argmax₂]]
            h, m = findmax([max₁, max₂])
            o[i, j] = h
            indexes[i, j] = argmaxmins[m]
        end
    end
    return o, indexes
end

function backward_block_segmented_hausdorff(ȳ, x, seg₁, seg₂, argmaxmins)
    o = zero(x)
    for (i, sᵢ) in enumerate(seg₁)
        for (j, sⱼ) in enumerate(seg₂)
            segment = o[sᵢ, sⱼ] 
            segment[argmaxmins[i, j]] += ȳ[i, j]
            o[sᵢ, sⱼ] .= segment # updated segment 
        end
    end
    return NoTangent(), o, NoTangent(), NoTangent(), NoTangent()
end