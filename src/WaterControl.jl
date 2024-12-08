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
export solve_forward
export create_animation
export create_surface_animation
export create_contour_animation

end
