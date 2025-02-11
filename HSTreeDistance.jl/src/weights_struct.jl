"""
# Differentable Dict-like struct

W = Dict{Any, Any}("Kg" => 1.0f0, "2" => 1.0f0, "6" => 1.0f0, "KE" => 1.0f0, "L*" => 1.0f0, "Kd" => 1.0f0, "A" => 1.0f0, "Ke" => 1.0f0, "E" => 1.0f0, "Kf" => 1.0f0, "I" => 1.0f0, "KU" => 1.0f0, "Kk" => 1.0f0)
ws = WeightStruct(W)

grad = gradient(
    ()->begin
        sum(map((k,y)->ws[k]*y, ws.keys, 1:13))
    end,
    Flux.params(ws)     
) 
"""

struct WeightStruct
    keys
    values
    transform::Function # Transformation of weights 
end

WeightStruct(x::Dict, transfrom::Function=identity) = WeightStruct(collect(keys(x)), Float32.(collect(values(x))), transform)
WeightStruct(x::NamedTuple, transform::Function=identity) = WeightStruct(collect(keys(x)), Float32.(collect(values(x))), transform)

Flux.@functor WeightStruct

Flux.trainable(ws::WeightStruct) = (values=ws.values,)

function Base.getindex(ws::WeightStruct, k::Union{String, Symbol})
    idx = Zygote.@ignore(findmax(ws.keys.==k)[2])
    return ws.transform.(ws.values[idx])
end

function destructure_metric_to_ws(m)
	buffer = []
	fmap(m; exclude=x->x isa WeightStruct) do x
		push!(buffer, x.keys)
	end;
	vcat(vec.(buffer)...), buffer
end 