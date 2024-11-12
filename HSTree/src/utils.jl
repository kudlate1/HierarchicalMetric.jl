function zerocardinality(bag)
    if isempty(bag)
        return true
    elseif length(bag) == 1 && length(bag[1]) == 0
        return true
    else
        return false
    end
end


function preprocess_missing(x::Array{<:Union{Missing, Number}}) #TODO make it to ignore gradients
    x̂ = copy(x);
    x̂[ismissing.(x)] .= 0;
    return x̂ # FIXME change type to Array{T}
end


function preprocess_missing(x::MaybeHotMatrix) #TODO make it to ignore gradients
    xsize = size(x);
    x̂ = copy(x);
    x̂[:, ismissing.(x[1,:])] .= onehot(xsize[1], 1:xsize[1]);
    return x̂ # FIXME change type to Array{OneHotArray}
end

# Temporary fix -- preprocessing function
## solves the problem with "1-lvl bags" with multiple empty observations
## TODO check if it work with "higher-lvl bags" 
#preprocess_missing(x.data)
preprocess_empty_bags(x::ArrayNode; extend::Bool=false, top_pn::Bool=false) = (extend && !top_pn) ? ArrayNode(hcat(x.data, zeros_like(x.data, (size(x.data,1), 1)))) : x

function preprocess_empty_bags(x::ProductNode{<:NamedTuple{M}}; extend::Bool=false, top_pn::Bool=false) where M 
    ms = map(key->preprocess_empty_bags(x.data[key], extend=extend, top_pn=top_pn), M)
    ProductNode(NamedTuple{M}(ms), x.metadata)
end

function preprocess_empty_bags(x::BagNode; extend::Bool=false, top_pn::Bool=false)
    #@show propertynames(x)
    #printtree(x)
    zc = zerocardinality.(x.bags)
    #@show isempty(zc)

    if isempty(zc)
        max_ = 1  # TODO check
        #@show "if", max_
    elseif length(zc) == 1 && sum(zc) == 1
        max_ = x.bags[1].stop + 1 #TODO test me properly
        #@show "elseif", max_
    else
        max_ = x.bags[.~zc][end].stop
        #@show "else", max_
    end

    if any(zc) || isempty(zc)
        if x.bags == AlignedBags([0:-1])
            new_bags = [range(max_+1, max_+1)]
        elseif isempty(x.bags)
            new_bags = [1:1]
        else
            new_bags = [(bag == 0:-1) ? range(max_+1, max_+1) : bag for bag in x.bags]
            new_bags = (extend && !top_pn) ? [new_bags..., range(max_+1, max_+1)] : new_bags
        end
        BagNode(preprocess_empty_bags(x.data; extend=true), new_bags, x.metadata)
    else #(extend && !top_pn) 
        new_bags = (extend && !top_pn) ? [x.bags..., range(max_+1, max_+1)] : x.bags # TODO add check to top_pn
        BagNode(preprocess_empty_bags(x.data; extend=extend, top_pn=false), new_bags, x.metadata)
    end
end

pad_leaves_for_wasserstein(x) = preprocess_empty_bags(x; extend=true, top_pn=true)