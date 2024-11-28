module WaterControl

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
export solve_forward_2D
export create_animation_1D
export create_surface_animation_2D
export create_contour_animation_2D

end
