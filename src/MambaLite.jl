module MambaLite

#################### Imports ####################

using Graphs: DiGraph
import LinearAlgebra: dot
using Showoff: showoff # Useful in hpd function


#################### Types ####################

ElementOrVector{T} = Union{T, Vector{T}}


#################### Variate Types ####################

abstract type ScalarVariate <: Real end
abstract type ArrayVariate{N} <: DenseArray{Float64, N} end

const AbstractVariate = Union{ScalarVariate, ArrayVariate}
const VectorVariate = ArrayVariate{1}
const MatrixVariate = ArrayVariate{2}


#################### Sampler Types ####################

mutable struct Sampler{T}
    params::Vector{Symbol}
    eval::Function
    tune::T
    targets::Vector{Symbol}
end

abstract type SamplerTune end

struct SamplerVariate{T<:SamplerTune} <: VectorVariate
    value::Vector{Float64}
    tune::T

    function SamplerVariate{T}(x::AbstractVector, tune::T) where T<:SamplerTune
      v = new{T}(x, tune)
      validate(v)
    end

    function SamplerVariate{T}(x::AbstractVector, pargs...; kargs...) where T<:SamplerTune
      value = convert(Vector{Float64}, x)
      new{T}(value, T(value, pargs...; kargs...))
    end
end


#################### Model Types ####################

struct ModelGraph
    graph::DiGraph
    keys::Vector{Symbol}
end

struct ModelState
    value::Vector{Float64}
    tune::Vector{Any}
end

mutable struct Model
    nodes::Dict{Symbol, Any}
    samplers::Vector{Sampler}
    states::Vector{ModelState}
    iter::Int
    burnin::Int
    hasinputs::Bool
    hasinits::Bool
end


#################### Chains Type ####################

abstract type AbstractChains end

struct Chains <: AbstractChains
    value::Array{Float64, 3}
    range::StepRange{Int, Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
end

struct ModelChains <: AbstractChains
    value::Array{Float64, 3}
    range::StepRange{Int, Int}
    names::Vector{AbstractString}
    chains::Vector{Int}
    model::Model
end


#################### Includes ####################

include("utils.jl")
include("variate.jl")

include("model/model.jl")

include("output/chains.jl")
include("output/chainsummary.jl")
include("output/stats.jl")

include("samplers/sampler.jl")

include("samplers/nuts.jl")


#################### Exports ####################

export Chains
export NUTS, NUTSVariate
export sample!

end # module MambaLite
