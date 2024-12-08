using LinearAlgebra
using Trapz


function jacobian!(j, h, u, prob::ShallowWaterProblem1D)
    nx = prob.nx
    dx = prob.dx
    μ = prob.μ
    ν = prob.ν
    g = 9.81

    fill!(j, 0.0)

    # boundary
    j[1, 1] = -u[2] / 2dx - μ / dx^2
    j[1, 2] = -u[1] / 2dx + μ / dx^2
    j[1, nx+1] = -h[2] / 2dx
    j[1, nx+2] = -h[1] / 2dx
    j[nx, nx-1] = u[nx] / 2dx + μ / dx^2
    j[nx, nx] = u[nx-1] / 2dx - μ / dx^2
    j[nx, 2nx-1] = h[nx] / 2dx
    j[nx, 2nx] = h[nx-1] / 2dx

    j[nx+1, 1] = g / 2dx
    j[nx+1, 2] = -g / 2dx
    j[nx+1, nx+1] = -u[1] / dx - 3ν / dx^2
    j[nx+1, nx+2] = -u[1] / 2dx + ν / dx^2
    j[2nx, nx-1] = g / dx
    j[2nx, nx] = -g / dx
    j[2nx, 2nx-1] = u[nx] / 2dx + ν / dx^2
    j[2nx, 2nx] = u[nx] / dx - 3ν / dx^2


    #bulk
    for i in 2:nx-1
        # ∂ḣ_i  / ∂h_j
        j[i, i-1] = -u[i] / 2dx + μ / dx^2
        j[i, i] = -(u[i+1] - u[i-1]) / 2dx - 2μ / dx^2
        j[i, i+1] = u[i] / 2dx + μ / dx^2

        # ∂ḣ_i  / ∂u_j
        j[i, nx+i-1] = h[i] / 2dx
        j[i, nx+i] = -(h[i+1] - h[i-1]) / 2dx
        j[i, nx+i+1] = -h[i] / 2dx
    end

    for i in nx+2:2nx-1
        # ∂u̇_i / ∂h_j
        j[i, i-1-nx] = g / 2dx
        j[i, i+1-nx] = -g / 2dx

        # ∂u̇_i / ∂u_j
        j[i, i-1] = u[i-nx] / 2dx + ν / dx^2
        j[i, i] = -(u[i+1-nx] - u[i-1-nx]) / 2dx - 2ν / dx^2
        j[i, i+1] = -u[i-nx] / 2dx + ν / dx^2
    end

    return j
end


function solve_adjoint(sol_forward, prob::ShallowWaterProblem1D, acc::DiscreteAcceleration1D)
    nx = prob.nx
    λ0 = vcat((sol_forward.u[end][1:nx] - prob.target), zeros(nx))
    tspan_adj = (sol_forward.t[end], sol_forward.t[1])
    
    # Create sufficiently dense timepoints for later integration
    t_dense = range(tspan_adj[1], tspan_adj[2], length=length(acc.values) * 50)
    
    function adjoint_with_prob!(dλ, λ, p, t)
        state = p(t)
        h = @view state[1:nx]
        u = @view state[nx+1:2nx]
        j = jacobian!(zeros(2nx, 2nx), h, u, prob)
        mul!(dλ, transpose(j), λ, -1.0, 0.0)
    end
    
    prob_adj = ODEProblem(adjoint_with_prob!, λ0, tspan_adj, sol_forward)
    return solve(prob_adj, Tsit5(), saveat=t_dense)
end


function solve_adjoint(sol_forward, prob::ShallowWaterProblem2D, acc::DiscreteAcceleration2D)
    nx, ny = prob.nx, prob.ny
    dx, dy = prob.dx, prob.dy
    g, ν, μ = 9.81, prob.ν, prob.μ
    
    # Reuse derivative matrices from forward solve
    D1h_x = WaterControl.create_first_spatial_derivative_matrix(nx, dx, false)
    D1h_y = WaterControl.create_first_spatial_derivative_matrix(ny, dy, false)
    D1uv_x = WaterControl.create_first_spatial_derivative_matrix(nx, dx, true)
    D1uv_y = WaterControl.create_first_spatial_derivative_matrix(ny, dy, true)

    D2h_x = WaterControl.create_second_spatial_derivative_matrix(nx, dx, false)
    D2h_y = WaterControl.create_second_spatial_derivative_matrix(ny, dy, false)
    D2uv_x = WaterControl.create_second_spatial_derivative_matrix(nx, dx, true)
    D2uv_y = WaterControl.create_second_spatial_derivative_matrix(ny, dy, true)
    
    # For storing intermediate derivatives
    dλh_dx = zeros(nx * ny)  # Result of D1h_x × λh
    dλu_dx = zeros(nx * ny)
    dλv_dx = zeros(nx * ny)
    dλh_dy = zeros(nx * ny)
    dλu_dy = zeros(nx * ny)
    dλv_dy = zeros(nx * ny)

    du_dx = zeros(nx * ny)
    dv_dy = zeros(nx * ny)

    d2λh_dx2 = zeros(nx * ny)
    d2λu_dx2 = zeros(nx * ny)
    d2λh_dy2 = zeros(nx * ny)
    d2λv_dy2 = zeros(nx * ny)
    
    function adjoint_rhs!(dλ, λ, p, t)
        state = p(t)
        h = @view state[1:nx*ny]
        u = @view state[nx*ny+1:2nx*ny]
        v = @view state[2nx*ny+1:3nx*ny]
        
        λh = @view λ[1:nx*ny]
        λu = @view λ[nx*ny+1:2nx*ny]
        λv = @view λ[2nx*ny+1:3nx*ny]

        dλh = @view dλ[1:nx*ny]
        dλu = @view dλ[nx*ny+1:2nx*ny]
        dλv = @view dλ[2nx*ny+1:3nx*ny]
        
        for i in 1:ny  # column-wise
            row_slice = i:ny:(nx*ny)

            du_dx_view = @view du_dx[row_slice]
            u_view = @view u[row_slice]
            mul!(du_dx_view, D1uv_x, u_view)

            dλh_dx_view = @view dλh_dx[row_slice]
            λh_view = @view λh[row_slice]
            mul!(dλh_dx_view, D1h_x, λh_view)

            dλu_dx_view = @view dλu_dx[row_slice]
            λu_view = @view λu[row_slice]
            mul!(dλu_dx_view, D1uv_x, λu_view)

            dλv_dx_view = @view dλv_dx[row_slice]
            λv_view = @view λv[row_slice]
            mul!(dλv_dx_view, D1uv_x, λv_view)

            d2λh_dx2_view = @view d2λh_dx2[row_slice]
            λh_view = @view λh[row_slice]
            mul!(d2λh_dx2_view, D2h_x, λh_view)

            d2λu_dx2_view = @view d2λu_dx2[row_slice]
            λu_view = @view λu[row_slice]
            mul!(d2λu_dx2_view, D2uv_x, λu_view)
        end

        for j in 1:nx
            col_slice = (j-1)*ny+1:j*ny

            dv_dy_view = @view dv_dy[col_slice]
            v_view = @view v[col_slice]
            mul!(dv_dy_view, D1uv_y, v_view)

            dλh_dy_view = @view dλh_dy[col_slice]
            λh_view = @view λh[col_slice]
            mul!(dλh_dy_view, D1h_y, λh_view)

            dλu_dy_view = @view dλu_dy[col_slice]
            λu_view = @view λu[col_slice]
            mul!(dλu_dy_view, D1uv_y, λu_view)

            dλv_dy_view = @view dλv_dy[col_slice]
            λv_view = @view λv[col_slice]
            mul!(dλv_dy_view, D1uv_y, λv_view)

            d2λh_dy2_view = @view d2λh_dy2[col_slice]
            λh_view = @view λh[col_slice]
            mul!(d2λh_dy2_view, D2h_y, λh_view)
            
            d2λv_dy2_view = @view d2λv_dy2[col_slice]
            λv_view = @view λv[col_slice]
            mul!(d2λv_dy2_view, D2uv_y, λv_view)
        end
        
        @. dλh = -(h * dλu_dx + u * dλh_dx + h * dλv_dy + v * dλh_dy) - μ * (d2λh_dx2 + d2λh_dy2)
        @. dλu = -(g * dλh_dx + u * dλu_dx + λu * du_dx + v * dλu_dy) - ν * d2λu_dx2
        @. dλv = -(g * dλh_dy + u * dλv_dx + λv * dv_dy + v * dλv_dy) - ν * d2λv_dy2
    end
    
    λ0 = vcat(sol_forward.u[end][1:nx*ny] - prob.target, zeros(2*nx*ny))
    tspan_adj = (sol_forward.t[end], sol_forward.t[1])
    prob_adj = ODEProblem(adjoint_rhs!, λ0, tspan_adj, sol_forward)
    t_dense = range(tspan_adj[1], tspan_adj[2], length=length(acc.times) * 10)

    solve(prob_adj, Tsit5(), saveat=t_dense)
end


function compute_control_gradient(sol_adj, t_intervals, nx)
    n_intervals = length(t_intervals)-1
    grad = zeros(n_intervals)
    
    for i in 1:n_intervals
        t_start, t_end = t_intervals[i], t_intervals[i+1]
        # Get indices directly from solution timepoints
        idx = findall(t -> t_start <= t <= t_end, sol_adj.t)
        ts = sol_adj.t[idx]
        integrand = [sum(view(sol_adj(t), nx+1:2nx)) for t in ts]
        grad[i] = trapz(ts, integrand)
    end
    return grad
end


function compute_control_gradient(sol_adj, t_intervals, nx, ny)
    n_intervals = length(t_intervals)-1
    grad_x = zeros(n_intervals)
    grad_y = zeros(n_intervals)
    
    for i in 1:n_intervals
        t_start, t_end = t_intervals[i], t_intervals[i+1]
        idx = findall(t -> t_start <= t <= t_end, sol_adj.t)
        ts = sol_adj.t[idx]

        # Sum λu for x-acceleration gradient
        integrand_x = [sum(view(sol_adj(t), nx*ny+1:2*nx*ny)) for t in ts]
        grad_x[i] = trapz(ts, integrand_x)
        
        # Sum λv for y-acceleration gradient
        integrand_y = [sum(view(sol_adj(t), 2*nx*ny+1:3*nx*ny)) for t in ts]
        grad_y[i] = trapz(ts, integrand_y)
    end
    return grad_x ./ 9, grad_y ./9   # not sure why we have this factor of 9
end


function update_control(acc::DiscreteAcceleration1D, grad, α=1.0)
    new_values = acc.values .- α * grad
    return DiscreteAcceleration1D(new_values, acc.times)
end


function update_control(acc::DiscreteAcceleration2D, grad_x, grad_y, α=1.0)
    new_x_values = acc.values_x .- α .* grad_x
    new_y_values = acc.values_y .- α .* grad_y
    return DiscreteAcceleration2D(new_x_values, new_y_values, acc.times)
end


function objective(sol, target)
    l = length(target)
    h_final = sol.u[end][1:l]
    return sum((h_final .- target).^2) / 2
end


function adjoint_gradient(prob::ShallowWaterProblem1D, acc::DiscreteAcceleration1D)
    # Solve forward in time
    sol = solve_forward(prob, acc)
    
    # Solve adjoint equation
    sol_adj = solve_adjoint(sol, prob, acc)
    
    # Compute gradient
    return compute_control_gradient(sol_adj, acc.times, prob.nx)
end


function adjoint_gradient(prob::ShallowWaterProblem2D, acc::DiscreteAcceleration2D)
    sol = solve_forward(prob, acc)
    sol_adj = solve_adjoint(sol, prob, acc)
    return compute_control_gradient(sol_adj, acc.times, prob.nx, prob.ny)
 end


function optimize(prob::ShallowWaterProblem2D)
    max_time = prob.tspan[2]
    acc_points = 50
    acc = DiscreteAcceleration2D(
        zeros(acc_points),
        zeros(acc_points),
        collect(LinRange(0, max_time, acc_points + 1))
    )

    current_err = 1
    α = 1e-4
    min_α = 1e-7  # Prevent step size from getting too small
    
    while current_err > 0.001
        sol = solve_forward(prob, acc)
        current_err = objective(sol, prob.target)
        println("Objective: $current_err")

        sol_adj = solve_adjoint(sol, prob, acc)
        grads = compute_control_gradient(sol_adj, acc.times, prob.nx, prob.ny)
        acc_new = update_control(acc, grads..., α)
        
        f_new = objective(solve_forward(prob, acc_new), prob.target)
        while f_new > current_err && α > min_α
            α *= 0.8
            acc_new = update_control(acc, grads..., α)
            f_new = objective(solve_forward(prob, acc_new), prob.target)
        end
        
        if α <= min_α && f_new > current_err
            break  # Exit if we can't make progress
        end
        
        acc = acc_new
        α = min(5α, 1e-2)
    end
    return acc
 end


function optimize(prob::ShallowWaterProblem1D)
    max_time = prob.tspan[2]
    acc_points = 50
    # acc = DiscreteAcceleration(0.1 * randn(acc_points), collect(LinRange(0, max_time, acc_points + 1)))
    acc = DiscreteAcceleration1D(zeros(acc_points), collect(LinRange(0, max_time, acc_points + 1)))
    
    current_err = 1
    α = 1.0
    min_α = 1e-7  # Prevent step size from getting too small
    
    while current_err > 0.001
        sol = solve_forward(prob, acc)
        current_err = objective(sol, prob.target)
        println("Objective: $current_err")

        grad = adjoint_gradient(prob, acc)
        acc_new = update_control(acc, grad, α)
        
        f_new = objective(solve_forward(prob, acc_new), prob.target)
        while f_new > current_err && α > min_α
            α *= 0.8
            acc_new = update_control(acc, grad, α)
            f_new = objective(solve_forward(prob, acc_new), prob.target)
        end
        
        if α <= min_α && f_new > current_err
            break  # Exit if we can't make progress
        end
        
        acc = acc_new
        α = min(5α, 1.0)
    end
    return acc
end
