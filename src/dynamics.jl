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


function solve_forward(problem::ShallowWaterProblem1D, acceleration::DiscreteAcceleration1D)
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


function solve_forward(problem::ShallowWaterProblem2D, acceleration::DiscreteAcceleration2D)
    nx = problem.nx
    ny = problem.ny
    dx = problem.dx
    dy = problem.dy
    ν = problem.ν
    μ = problem.μ
    g = 9.81

    D1h_x = create_first_spatial_derivative_matrix(nx, dx, false)
    D1h_y = create_first_spatial_derivative_matrix(ny, dy, false)
    D1uv_x = create_first_spatial_derivative_matrix(nx, dx, true)
    D1uv_y = create_first_spatial_derivative_matrix(ny, dy, true)

    D2h_x = create_second_spatial_derivative_matrix(nx, dx, false)
    D2h_y = create_second_spatial_derivative_matrix(ny, dy, false)
    D2uv_x = create_second_spatial_derivative_matrix(nx, dx, true)
    D2uv_y = create_second_spatial_derivative_matrix(ny, dy, true)

    dhdx = zeros(nx * ny)
    dhdy = zeros(nx * ny)
    dudx = zeros(nx * ny)
    dudy = zeros(nx * ny)
    dvdx = zeros(nx * ny)
    dvdy = zeros(nx * ny)

    d2hdx2 = zeros(nx * ny)
    d2hdy2 = zeros(nx * ny)
    d2udx2 = zeros(nx * ny)
    d2vdy2 = zeros(nx * ny)

    function shallow_water!(dw, w, p, t)
        h = @view w[1 : nx * ny]
        u = @view w[nx * ny + 1 : 2 * nx * ny]      # x velocity
        v = @view w[2 * nx * ny + 1 : 3 * nx * ny]  # y velocity
        
        dh = @view dw[1 : nx * ny]
        du = @view dw[nx * ny + 1 : 2 * nx * ny]      # x velocity
        dv = @view dw[2 * nx * ny + 1 : 3 * nx * ny]  # y velocity

        # for each row, compute x derivatives
        for i in 1:ny
            dhdx_view = @view dhdx[i:ny:end]
            h_view = @view h[i:ny:end]
            mul!(dhdx_view, D1h_x, h_view)

            dudx_view = @view dudx[i:ny:end]
            u_view = @view u[i:ny:end]
            mul!(dudx_view, D1uv_x, u_view)

            dvdx_view = @view dvdx[i:ny:end]
            v_view = @view v[i:ny:end]
            mul!(dvdx_view, D1uv_x, v_view)

            d2hdx2_view = @view d2hdx2[i:ny:end]
            mul!(d2hdx2_view, D2h_x, h_view)

            d2udx2_view = @view d2udx2[i:ny:end]
            mul!(d2udx2_view, D2uv_x, u_view)
        end

        # for each column, compute y derivatives
        for j in 1:nx
            dhdy_view = @view dhdy[(j-1)*ny+1:j*ny]
            h_view = @view h[(j-1)*ny+1:j*ny]
            mul!(dhdy_view, D1h_y, h_view)

            dudy_view = @view dudy[(j-1)*ny+1:j*ny]
            u_view = @view u[(j-1)*ny+1:j*ny]
            mul!(dudy_view, D1uv_y, u_view)

            dvdy_view = @view dvdy[(j-1)*ny+1:j*ny]
            v_view = @view v[(j-1)*ny+1:j*ny]
            mul!(dvdy_view, D1uv_y, v_view)

            d2hdy2_view = @view d2hdy2[(j-1)*ny+1:j*ny]
            mul!(d2hdy2_view, D2h_y, h_view)

            d2vdy2_view = @view d2vdy2[(j-1)*ny+1:j*ny]
            mul!(d2vdy2_view, D2uv_y, v_view)
        end

        ax, ay = acceleration(t)

        @. dh = -h * dudx - u * dhdx - h * dvdy - v * dhdy + μ * (d2hdx2 + d2hdy2)
        @. du = -ax - g * dhdx - u * dudx - v * dudy + ν * d2udx2
        @. dv = -ay - g * dhdy - u * dvdx - v * dvdy + ν * d2vdy2
    end

    # Initial conditions
    h0 = ones(nx * ny)
    u0 = zeros(nx * ny)
    v0 = zeros(nx * ny)
    w0 = vcat(h0, u0, v0)

    # Create and solve the problem
    prob = ODEProblem(shallow_water!, w0, problem.tspan)
    return solve(prob, Tsit5(), tstops=acceleration.times)
end
