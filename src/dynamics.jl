using DifferentialEquations
using LinearAlgebra


function create_first_spatial_derivative_matrix(nx::Int, dx::Float64, is_velocity::Bool = false)
    # Create a matrix with 1/2dx on the superdiagonal and -1/2dx on the subdiagonal
    D1 = Tridiagonal(
        fill(-1/(2dx), nx-1),  # subdiagonal
        zeros(nx),             # diagonal
        fill(1/(2dx), nx-1)    # superdiagonal
    )
    
    # Modify the corner cases
    if is_velocity
        D1[1,1] = 1 / 2dx      # top left
        D1[nx,nx] = -1 / 2dx   # bottom right
    else
        D1[1,1] = -1 / 2dx     # top left
        D1[nx,nx] = 1 / 2dx    # bottom right
    end

    return D1
end


function create_second_spatial_derivative_matrix(nx::Int, dx::Float64, is_velocity::Bool = false)
    D2 = Tridiagonal(
        fill(1 / dx^2, nx-1),   # subdiagonal
        fill(-2 / dx^2, nx),    # diagonal
        fill(1 / dx^2, nx-1)    # superdiagonal
    )
    
    # Modify the corner entries
    if is_velocity
        D2[1,1] = -3 / dx^2     # top left
        D2[nx,nx] = -3 / dx^2   # bottom right
    else
        D2[1,1] = -1 / dx^2     # top left
        D2[nx,nx] = -1 / dx^2   # bottom right
    end

    return D2
end


function solve_forward_1D(problem::ShallowWaterProblem1D, acceleration::DiscreteAcceleration1D)
    # Modified shallow water equations with container motion
    nx = problem.nx
    dx = problem.dx
    ν = problem.ν
    μ = problem.μ
    g = 9.81

    D1h = create_first_spatial_derivative_matrix(nx, dx, false)
    D1u = create_first_spatial_derivative_matrix(nx, dx, true)
    D2h = create_second_spatial_derivative_matrix(nx, dx, false)
    D2u = create_second_spatial_derivative_matrix(nx, dx, true)

    # Preallocate for matrix multiplications
    temp_v = zeros(nx)
    temp_h = zeros(nx)

    function shallow_water!(du, u, p, t)
        h = @view u[1:nx]
        v = @view u[nx+1:2nx]
        dh = @view du[1:nx]
        dv = @view du[nx+1:2nx]

        mul!(temp_h, D1h, h)
        mul!(temp_v, D1u, v)
        
        @. dh = -h * temp_v - v * temp_h
        @. dv = -acceleration(t) - v * temp_v - g * temp_h

        mul!(temp_h, D2h, h)
        mul!(temp_v, D2u, v)

        @. dh += μ * temp_h
        @. dv += ν * temp_v
    end

    u0 = vcat(problem.h0, problem.v0)
    prob = ODEProblem(shallow_water!, u0, problem.tspan)
    return solve(prob, Tsit5(), tstops=acceleration.times)
end
