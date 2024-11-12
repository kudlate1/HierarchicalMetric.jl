abstract type AbstractMetric end

const NTup{T} = NamedTuple{K, <:Tuple{Vararg{T}}} where K

# LeafMetric
struct LeafMetric <: AbstractMetric
    metric
    type
    keyname
end

LeafMetric(metric) = LeafMetric(metric, "Any", "0")

Flux.@functor LeafMetric

function (m::LeafMetric)(x::ArrayNode, y::ArrayNode)
    if isempty(x.data) && isempty(y.data)
        0
    elseif isempty(y.data) 
        m.metric(x.data, zeros_like(x.data, (size(x.data,1), 1))) #TODO thik of better solution
    elseif isempty(x.data)
        m.metric(zeros_like(x.data, (size(x.data,1), 1)), y.data) #TODO thik of better solution 
    else
        m.metric(x.data, y.data)
    end
end

# TODO version for KnowledgeBase

# ProductMetric
struct ProductMetric{T<:NamedTuple, U, W<:Union{WeightStruct, NamedTuple, Vector}} <: AbstractMetric
    ms::T # instance metrices
    pm::U # product metric
    weights::W
    keyname
end

Flux.@functor ProductMetric

function (m::ProductMetric{<:NamedTuple{M}})(x::ProductNode{<:NamedTuple{X}}, y::ProductNode{<:NamedTuple{Y}}) where {M,X,Y} 
    @assert issubset(M, X) && issubset(M, Y)
    elements = map(M) do k
        m.ms[k](x.data[k], y.data[k])
    end
    product = cat(elements..., dims=3)
    weights_ = reshape(collect(map(k->m.weights[k], M)), 1,1,:)
    m.pm(product, weights_)
end

# TODO version for KnowledgeBase

"""
    SetMetric <: AbstractMetric
"""
struct SetMetric{T<:AbstractMetric, S, C} <: AbstractMetric
    im::T # instance metric
    sm::S # set metric
    cm::C # cardinality metric (scale * sm + bias)
    keyname
end

Flux.@functor SetMetric

# this work for comparision of single "empty bag" with "nonepty bags" i.e. [0:-1] vs [1:1, 2:3, 3:5] 
function (m::SetMetric)(x::BagNode, y::BagNode) # FIXME
    if all(zerocardinality.([x.bags, y.bags]))  return zeros(Float32, 1,1) end
    elements = m.im(x.data, y.data)
    xbags = zerocardinality(x.bags) ? AlignedBags(1:1) : x.bags
    ybags = zerocardinality(y.bags) ? AlignedBags(1:1) : y.bags
    c_scale, c_bias = m.cm(xbags, ybags)
    bsn = block_segmented_norm(elements, xbags, ybags, m.sm)
    c_scale .* bsn .+ c_bias
end

# TODO version for KnowledgeBase