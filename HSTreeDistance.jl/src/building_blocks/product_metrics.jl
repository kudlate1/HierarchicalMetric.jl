WeightedProductMetric(p,w) = dropdims(sqrt.(sum(w .* (p.^2), dims=3) .+ 1f-20), dims=3)
SqWeightedProductMetric(p,w) = dropdims(sum(w .* p, dims=3), dims=3)