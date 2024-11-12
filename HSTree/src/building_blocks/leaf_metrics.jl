"""
dist_(x::Array, y::Array) = sqrt.(Distances.pairwise(Distances.SqEuclidean(), x, y))
dist_(x::Flux.OneHotArray, y::Flux.OneHotArray) = Distances.pairwise(Distances.Cityblock(), x, y) ./2 
dist_(x::ArrayNode{<:NGramMatrix{String}}, y::ArrayNode{<:NGramMatrix{String}}) = pairwise(NormLevenshtein, x.data.S, y.data.S)
"""

# Special version for String data
NormLevenshtein(x,y) = Levenshtein()(x,y)/min(length(x), length(y))
Pairwise_Levenstein(x::ArrayNode{<:NGramMatrix{String}}, y::ArrayNode{<:NGramMatrix{String}}) = pairwise(NormLevenshtein, x.data.S, y.data.S)

# Numerical data
Pairwise_Euclidean(x::Array, y::Array) = sqrt.(Distances.pairwise(Distances.SqEuclidean(), x, y) .+ 1f-20)
Pairwise_SqEuclidean(x::Array, y::Array) = Distances.pairwise(Distances.SqEuclidean(), x, y)

# Categorical data
#TODO fix this # if one is missing loss is 0.5 not 1
Pairwise_Cityblock(x::Flux.OneHotArray, y::Flux.OneHotArray) = Distances.pairwise(Distances.SqEuclidean(), x, y) ./2 # equivalent to Cityblock distance
Pairwise_Cityblock(x::Flux.OneHotArray, y::Matrix{Bool}) = Distances.pairwise(Distances.SqEuclidean(), x, y) ./2
Pairwise_Cityblock(x::Matrix{Bool}, y::Flux.OneHotArray) = Distances.pairwise(Distances.SqEuclidean(), x, y) ./2
Pairwise_Cityblock(x::Matrix{Bool}, y::Matrix{Bool}) = Distances.pairwise(Distances.SqEuclidean(), x, y) ./2
#Pairwise_Cityblock(x::Matrix, y::Matrix) = Distances.pairwise(Distances.SqEuclidean(), x, y) ./2



# version for missing data
# Probably gradients are not going to work # missing and Number
#=
function Pairwise_Euclidean(x::Array{<:Union{Missing, Number}}, y::Array{<:Union{Missing, Number}})
    x̂ = copy(x); x̂[ismissing.(x̂)] .= 0
    ŷ = copy(y); ŷ[ismissing.(ŷ)] .= 0
    return Pairwise_Euclidean(x̂, ŷ)
end


# MaybeHotMatrix # TODO think about creating separate token for missing already in extractors!
function Pairwise_Cityblock(x::MaybeHotMatrix, y::MaybeHotMatrix) 
    xsize, ysize = size(x), size(y);
    x̂ = copy(x); 
    ŷ = copy(y); 
    x̂[:,ismissing.(x[1,:])] .= onehot(xsize[1], 1:xsize[1]); # using OneHotArrays
    ŷ[:,ismissing.(y[1,:])] .= onehot(ysize[1], 1:ysize[1]); # using OneHotArrays

    return Pairwise_Cityblock(x̂, ŷ)    
end
=#