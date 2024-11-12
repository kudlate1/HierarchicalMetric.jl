# function for reflecting metric from data/ schema
function reflectmetric(
    x; 
    prod_metric=WeightedProductMetric, 
    set_metric=ChamferDistance, 
    leaf_metrics=Dict(
        :continuous => Pairwise_Euclidean,
        :categorical => Pairwise_Cityblock,
        :string => Pairwise_Levenstein
    ), 
    card_metric=ScaleOne,
    weight_sampler=ones, 
    weight_transform=identity,
    s="0"
)
    return _reflectmetric(x, prod_metric, set_metric, leaf_metrics, card_metric, weight_sampler, weight_transform, s)
end

# TODO add maybe oh / missing / 

function _reflectmetric(x::ArrayNode{<:Array}, prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s)
    return LeafMetric(leaf_matric[:continuous], "con", Mill.stringify(s))
end

function _reflectmetric(x::ArrayNode{<:Flux.OneHotArray}, prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s) 
    return LeafMetric(leaf_matric[:categorical], "cat", Mill.stringify(s))
end

function _reflectmetric(x::ArrayNode{<:NGramMatrix}, prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s) 
    return LeafMetric(leaf_matric[:string], "str", Mill.stringify(s))
end


function _reflectmetric(x::AbstractProductNode, prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s)
    c = Mill.stringify(s)
    n = length(x.data)
    ks = keys(x.data)
    ms = [_reflectmetric(x.data[k], prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s * Mill.encode(i, n))
                  for (i, k) in enumerate(ks)]
    weights = WeightStruct(Mill._remap(x.data, weight_sampler(length(ms))), weight_transform)
    ms = Mill._remap(x.data, ms)
    return ProductMetric(ms, prod_metric, weights, c)
end

function _reflectmetric(x::AbstractBagNode, prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s)
    c = Mill.stringify(s)
    im = _reflectmetric(x.data, prod_metric, set_metric, leaf_matric, card_metric, weight_sampler, weight_transform, s * Mill.encode(1, 1))
    return SetMetric(im, set_metric, card_metric, c)
end