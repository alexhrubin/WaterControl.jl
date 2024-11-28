module ShallowWater

include("types.jl")
include("dynamics.jl")
include("adjoint.jl")
include("visualization.jl")

export ShallowWaterProblem
export DiscreteAcceleration
export optimize
export solve_forward
export create_animation

end
