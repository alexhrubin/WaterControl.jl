module WaterControl

include("types.jl")
include("dynamics.jl")
include("adjoint.jl")
include("visualization.jl")
include("mesh.jl")
include("galerkin.jl")
include("relax.jl")

export ShallowWaterProblem1D
export ShallowWaterProblem2D
export DiscreteAcceleration1D
export DiscreteAcceleration2D
export optimize
export solve_forward
export create_animation
export create_surface_animation
export create_contour_animation
export make_mesh, dual_mesh_values
export Mesh, Point, HalfEdge, Face
export solve_poisson_fem
export solve_poisson_sor

end
