"""
As cardinality metric can be generally used anything function with two outputs with correct dimensions.

Cardinality metrics is aplied in SetMetric in following manner
    
    c_scale, c_bias = arbitrary_card_metric(xbags, ybags)
    output = c_scale .* bag_metric .+ c_bias

Thanks to that many versions of cardinality metrics can be used, i.e.

    1) # Normalized Wasserstein

        output = 1.0 .* Wₚ(x,y) .+ 0.0 
    
    2) # Unnormialized Wasserstein (for Mulitsets)
        (proposed in Chuang, Ch.: Tree Mover’s Distance: Bridging Graph Metrics and Stability of Graph Neural Networks 
            https://arxiv.org/abs/2210.01906)

        output = M .* Wₚ(x,y) .+ 0, where M = max(xcard, ycard)
    
    3) # Affine combination 
        (proposed in Bolt, G.: Distances for Comparing Multisets and Sequences -- https://arxiv.org/abs/2206.08858) 

        output = α .* Wₚ(x,y) .* (1-α) .* C(xcard, ycard), where α ∈ (0,1) and C is integer cardinality metric 
"""


ScaleOne(xbag, ybag) = (1f0, 0f0)


"""
    MaxCard(xbags, ybags)

Function that returns matrix of max cardinalites of two bags.

(proposed in Chuang, Ch.: Tree Mover’s Distance: Bridging Graph Metrics and Stability of Graph Neural Networks 
    https://arxiv.org/abs/2210.01906)


Effect on bag metric is following
```math
    y = m ⋅ Wₚ(x,y) + 0, where m = max(|x|, |y|)
```

"""
function MaxCard(xbags, ybags)
    card_prod = Zygote.@ignore collect(Base.Iterators.product(xbags, ybags))
    max_cards = Zygote.@ignore maximum.([(length(x), length(y)) for (x,y) in card_prod])
    return max_cards, 0f0
end
