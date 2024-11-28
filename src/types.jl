struct ShallowWaterProblem1D
    # Physical parameters
    L::Float64    # domain length
    nx::Int       # number of spatial points
    ν::Float64    # bulk viscosity coefficient
    μ::Float64    # artificial diffusion
    
    # Grid
    dx::Float64
    x::Vector{Float64}
    
    # Time settings
    tspan::Tuple{Float64,Float64}
    
    # Initial conditions
    h0::Vector{Float64}
    v0::Vector{Float64}
    
    # Target state
    target::Vector{Float64}
    
    # Constructor
    function ShallowWaterProblem1D(target::Vector{Float64};
            L=10.0, nx=100, ν=0.8, μ=0.1,
            tspan=(0.0,10.0))
        
        @assert length(target) == nx "Target length must match nx"
        
        dx = L/nx
        x = LinRange(0, L, nx)
        
        h0 = ones(nx)
        v0 = zeros(nx)
        
        new(L, nx, ν, μ,    # physical parameters
            dx, collect(x),  # grid (collect to make x a Vector)
            tspan,  # time settings
            h0, v0,         # initial conditions
            target)         # target
    end
end


mutable struct DiscreteAcceleration1D
    values::Vector{Float64}
    times::Vector{Float64}
    
    function DiscreteAcceleration1D(values, times)
        values_vec = collect(Float64, values)
        times_vec = collect(Float64, times)
        @assert length(values_vec) == length(times_vec)-1 "Need one fewer values than time points"
        new(values_vec, times_vec)
    end
end


# Interpolate acceleration as piecewise constant
function (a::DiscreteAcceleration1D)(t::Real)
    @assert t >= 0 throw(DomainError(6, "Input out of bounds"))

    i = findfirst(p -> p > t, a.times)
    if i === nothing  # t is beyond last point
        return a.values[end]
    end
    i = max(1, i-1)
    return a.values[i]
end


import Base: getindex, setindex!, lastindex

getindex(a::DiscreteAcceleration1D, i) = a.values[i];
setindex!(a::DiscreteAcceleration1D, v, i) = (a.values[i] = v);
lastindex(a::DiscreteAcceleration1D) = length(a.values)


mutable struct DiscreteAcceleration2D
    values_x::Vector{Float64}
    values_y::Vector{Float64}
    times::Vector{Float64}
    
    function DiscreteAcceleration2D(values_x, values_y, times)
        values_x_vec = collect(Float64, values_x)
        values_y_vec = collect(Float64, values_y)
        times_vec = collect(Float64, times)
        @assert length(values_x_vec) == length(times_vec)-1 "Need one fewer values than time points"
        @assert length(values_y_vec) == length(times_vec)-1 "Need one fewer values than time points"
        new(values_x_vec, values_y_vec, times_vec)
    end
end


getindex(a::DiscreteAcceleration2D, i) = a.values_x[i], a.values_y[i];
setindex!(a::DiscreteAcceleration2D, v::Tuple{Real,Real}, i) = (a.values_x[i] = v[1]; a.values_y[i] = v[2])
lastindex(a::DiscreteAcceleration2D) = length(a.values_x)


# Interpolate acceleration as piecewise constant
function (a::DiscreteAcceleration2D)(t::Real)
    @assert t >= 0 throw(DomainError(6, "Input out of bounds"))

    i = findfirst(p -> p > t, a.times)
    if i === nothing  # t is beyond last point
        return a[end]
    end
    i = max(1, i-1)
    return a[i]
end
