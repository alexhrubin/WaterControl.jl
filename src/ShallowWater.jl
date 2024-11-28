module ShallowWater

include("types.jl")
include("dynamics.jl")
include("adjoint.jl")
include("visualization.jl")

export ShallowWaterProblem1D
export ShallowWaterProblem2D
export DiscreteAcceleration1D
export DiscreteAcceleration2D
export optimize
export solve_forward_1D
export create_animation

end
