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


function solve_adjoint(sol_forward, prob, acc)
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


function update_control(acc, grad, α=1.0)
    new_values = acc.values .- α * grad
    return DiscreteAcceleration1D(new_values, acc.times)
end


function objective(sol, target)
    nx = length(target)
    h_final = sol.u[end][1:nx]
    return sum((h_final .- target).^2) / 2
end


function adjoint_gradient(prob, acc)
    # Solve forward in time
    sol = solve_forward_1D(prob, acc)
    
    # Solve adjoint equation
    sol_adj = solve_adjoint(sol, prob, acc)
    
    # Compute gradient
    return compute_control_gradient(sol_adj, acc.times, prob.nx)
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
        sol = solve_forward_1D(prob, acc)
        current_err = objective(sol, prob.target)
        println("Objective: $current_err")

        grad = adjoint_gradient(prob, acc)
        acc_new = update_control(acc, grad, α)
        
        f_new = objective(solve_forward_1D(prob, acc_new), prob.target)
        while f_new > current_err && α > min_α
            α *= 0.8
            acc_new = update_control(acc, grad, α)
            f_new = objective(solve_forward_1D(prob, acc_new), prob.target)
        end
        
        if α <= min_α && f_new > current_err
            break  # Exit if we can't make progress
        end
        
        acc = acc_new
        α = min(5α, 1.0)
    end
    return acc
end


function flat_idx(i::Int, j::Int, nx::Int)::Int
    return i + nx*(j-1)
end


function jacobian_2D!(j, h, u, v, prob::ShallowWaterProblem2D)
    fill!(j, 0.0)
    jacobian_2D_bulk_terms!(j, h, u, v, prob)
    jacobian_2D_boundary_terms!(j, h, u, v, prob)
    jacobian_2D_corner_terms!(j, h, u, v, prob)
    return j
end


function jacobian_2D_bulk_terms!(jac, h, u, v, prob::ShallowWaterProblem2D)
    nx = prob.nx
    ny = prob.ny
    dx = prob.dx
    dy = prob.dy
    μ = prob.μ
    ν = prob.ν
    g = 9.81

    for i in 2:ny-1
        for j in 2:nx-1
            k = flat_idx(i, j, nx)
            k_i_m = flat_idx(i-1, j, nx)
            k_i_p = flat_idx(i+1, j, nx)
            k_j_m = flat_idx(i, j-1, nx)
            k_j_p = flat_idx(i, j+1, nx)
     
            # For h[i, j] looking at h[_, _]:
     
            # ∂h[i, j]/∂h[i, -1 + j] = μ/Power(dx,2) + u(i,j)/(2.*dx)
            jac[k, k_j_m] = μ / dx^2 + u[k] / 2dx
     
            # ∂h[i, j]/∂h[i, j] = (-2*μ)/Power(dx,2) - (2*μ)/Power(dy,2) - (-u(i,-1 + j) + u(i,1 + j))/(2.*dx) - (-v(-1 + i,j) + v(1 + i,j))/(2.*dy)
            jac[k, k] = -2μ / dx^2 - 2μ / dy^2 - (-u[k_j_m] + u[k_j_p]) / 2dx - (-v[k_i_m] + v[k_i_p]) / 2dy
     
            # ∂h[i, j]/∂h[i, 1 + j] = μ/Power(dx,2) - u(i,j)/(2.*dx)
            jac[k, k_j_p] = μ / dx^2 - u[k] / 2dx
     
            # ∂h[i, j]/∂h[-1 + i, j] = μ/Power(dy,2) + v(i,j)/(2.*dy)
            jac[k, k_i_m] = μ / dy^2 + v[k] / 2dy
     
            # ∂h[i, j]/∂h[1 + i, j] = μ/Power(dy,2) - v(i,j)/(2.*dy)
            jac[k, k_i_p] = μ / dy^2 - v[k] / 2dy

            # For h[i, j] looking at u[_, _]:

            # ∂h[i, j]/∂u[i, j] = -0.5*(-h(i,-1 + j) + h(i,1 + j))/dx
            jac[k, k + nx*ny] = -(-h[k_j_m] + h[k_j_p]) / 2dx

            # ∂h[i, j]/∂u[i, -1 + j] = h(i,j)/(2.*dx)
            jac[k, k_j_m + nx*ny] = h[k] / 2dx

            # ∂h[i, j]/∂u[i, 1 + j] = -0.5*h(i,j)/dx
            jac[k, k_j_p + nx*ny] = -h[k] / 2dx

            # For h[i, j] looking at v[_, _]:

            # ∂h[i, j]/∂v[i, j] = -0.5*(-h(-1 + i,j) + h(1 + i,j))/dy
            jac[k, k + 2nx*ny] = -(-h[k_i_m] + h[k_i_p]) / 2dy

            # ∂h[i, j]/∂v[-1 + i, j] = h(i,j)/(2.*dy)
            jac[k, k_i_m + 2nx*ny] = h[k] / 2dy

            # ∂h[i, j]/∂v[1 + i, j] = -0.5*h(i,j)/dy
            jac[k, k_i_p + 2nx*ny] = -h[k] / 2dy


            # For u[i, j] looking at h[_, _]:
    
            # ∂u[i, j]/∂h[i, -1 + j] = g/(2.*dx)
            jac[k + nx*ny, k_j_m] = g / 2dx
    
            # ∂u[i, j]/∂h[i, 1 + j] = -0.5*g/dx
            jac[k + nx*ny, k_j_p] = -g / 2dx
    
            # For u[i, j] looking at u[_, _]:
    
            # ∂u[i, j]/∂u[i, j] = (-2*ν)/Power(dx,2) - (-u(i,-1 + j) + u(i,1 + j))/(2.*dx)
            jac[k + nx*ny, k + nx*ny] = -2ν / dx^2 - (-u[k_j_m] + u[k_j_p]) / 2dx
    
            # ∂u[i, j]/∂u[i, -1 + j] = ν/Power(dx,2) + u(i,j)/(2.*dx)
            jac[k + nx*ny, k_j_m + nx*ny] = ν / dx^2 + u[k] / 2dx
    
            # ∂u[i, j]/∂u[i, 1 + j] = ν/Power(dx,2) - u(i,j)/(2.*dx)
            jac[k + nx*ny, k_j_p + nx*ny] = ν / dx^2 - u[k] / 2dx
    
            # ∂u[i, j]/∂u[-1 + i, j] = v(i,j)/(2.*dy)
            jac[k + nx*ny, k_i_m + nx*ny] = v[k] / 2dy
    
            # ∂u[i, j]/∂u[1 + i, j] = -0.5*v(i,j)/dy
            jac[k + nx*ny, k_i_p + nx*ny] = -v[k] / 2dy
    
            # For u[i, j] looking at v[_, _]:
    
            # ∂u[i, j]/∂v[i, j] = -0.5*(-u(-1 + i,j) + u(1 + i,j))/dy
            jac[k + nx*ny, k + 2nx*ny] = -(-u[k_i_m] + u[k_i_p]) / 2dy


            # For v[i, j] looking at h[_, _]:
       
            # ∂v[i, j]/∂h[-1 + i, j] = g/(2.*dy)
            jac[k + 2nx*ny, k_i_m] = g / 2dy

            # ∂v[i, j]/∂h[1 + i, j] = -0.5*g/dy
            jac[k + 2nx*ny, k_i_p] = -g / 2dy

            # For v[i, j] looking at u[_, _]:

            # ∂v[i, j]/∂u[i, j] = -0.5*(-v(i,-1 + j) + v(i,1 + j))/dx
            jac[k + 2nx*ny, k + nx*ny] = -(-v[k_j_m] + v[k_j_p]) / 2dx

            # For v[i, j] looking at v[_, _]:

            # ∂v[i, j]/∂v[i, -1 + j] = u(i,j)/(2.*dx)
            jac[k + 2nx*ny, k_j_m + 2nx*ny] = u[k] / 2dx

            # ∂v[i, j]/∂v[i, 1 + j] = -0.5*u(i,j)/dx
            jac[k + 2nx*ny, k_j_p + 2nx*ny] = -u[k] / 2dx

            # ∂v[i, j]/∂v[i, j] = (-2*ν)/Power(dy,2) - (-v(-1 + i,j) + v(1 + i,j))/(2.*dy)
            jac[k + 2nx*ny, k + 2nx*ny] = -2ν / dy^2 - (-v[k_i_m] + v[k_i_p]) / 2dy

            # ∂v[i, j]/∂v[-1 + i, j] = ν/Power(dy,2) + v(i,j)/(2.*dy)
            jac[k + 2nx*ny, k_i_m + 2nx*ny] = ν / dy^2 + v[k] / 2dy

            # ∂v[i, j]/∂v[1 + i, j] = ν/Power(dy,2) - v(i,j)/(2.*dy)
            jac[k + 2nx*ny, k_i_p + 2nx*ny] = ν / dy^2 - v[k] / 2dy
        end
    end
end


function jacobian_2D_boundary_terms!(jac, h, u, v, prob::ShallowWaterProblem2D)
    nx = prob.nx
    ny = prob.ny
    dx = prob.dx
    dy = prob.dy
    μ = prob.μ
    ν = prob.ν
    g = 9.81

    ### hdot boundaries ###

    for j in 2:nx-1
        k = flat_idx(1, j, nx)
        k_i_p = flat_idx(2, j, nx)
        k_j_m = flat_idx(1, j-1, nx)
        k_j_p = flat_idx(1, j+1, nx)

        # For boundary case {i -> 1} looking at h[_, _]:

        # ∂h[1, j]/∂h[1, -1 + j] = μ/Power(dx,2) + u(1,j)/(2.*dx)
        jac[k, k_j_m] = u[k] / 2dx + μ / dx^2

        # ∂h[1, j]/∂h[1, j] = (-2*μ)/Power(dx,2) - μ/Power(dy,2) - (-u(1,-1 + j) + u(1,1 + j))/(2.*dx) + v(1,j)/(2.*dy) - (v(1,j) + v(2,j))/(2.*dy)
        jac[k, k] = -2μ / dx^2 - μ / dy^2 - (-u[k_j_m] + u[k_j_p]) / 2dx + v[k] / 2dy - (v[k] + v[k_i_p]) / 2dy

        # ∂h[1, j]/∂h[1, 1 + j] = μ/Power(dx,2) - u(1,j)/(2.*dx)
        jac[k, k_j_p] = μ / dx^2 - u[k] / 2dx

        # ∂h[1, j]/∂h[2, j] = μ/Power(dy,2) - v(1,j)/(2.*dy)
        jac[k, k_i_p] = μ / dy^2 - v[k] / 2dy

        # For boundary case {i -> 1} looking at u[_, _]:

        # ∂h[1, j]/∂u[1, j] = -0.5*(-h(1,-1 + j) + h(1,1 + j))/dx
        jac[k, k + nx*ny] = -(-h[k_j_m] + h[k_j_p]) / 2dx

        # ∂h[1, j]/∂u[1, -1 + j] = h(1,j)/(2.*dx)
        jac[k, k_j_m + nx*ny] = h[k] / 2dx

        # ∂h[1, j]/∂u[1, 1 + j] = -0.5*h(1,j)/dx
        jac[k, k_j_p + nx*ny] = -h[k] / 2dx

        # For boundary case {i -> 1} looking at v[_, _]:

        # ∂h[1, j]/∂v[1, j] = -0.5*h(1,j)/dy - (-h(1,j) + h(2,j))/(2.*dy)
        jac[k, k + 2nx*ny] = -h[k] / 2dy - (-h[k] + h[k_i_p]) / 2dy

        # ∂h[1, j]/∂v[2, j] = -0.5*h(1,j)/dy
        jac[k, k_i_p + 2nx*ny] = -h[k] / 2dy
    end

    for j in 2:nx-1
        k = flat_idx(ny, j, nx)
        k_i_m = flat_idx(ny-1, j, nx)
        k_j_m = flat_idx(ny, j-1, nx)
        k_j_p = flat_idx(ny, j+1, nx)
     
        # For boundary case {i -> n} looking at h[_, _]:
        
        # ∂h[n, j]/∂h[-1 + n, j] = μ/Power(dy,2) + v(n,j)/(2.*dy)
        jac[k, k_i_m] = μ / dy^2 + v[k] / 2dy
     
        # ∂h[n, j]/∂h[n, j] = (-2*μ)/Power(dx,2) - μ/Power(dy,2) - (-u(n,-1 + j) + u(n,1 + j))/(2.*dx) - (-v(-1 + n,j) - v(n,j))/(2.*dy) - v(n,j)/(2.*dy)
        jac[k, k] = -2μ / dx^2 - μ / dy^2 - (-u[k_j_m] + u[k_j_p]) / 2dx - (-v[k_i_m] - v[k]) / 2dy  - v[k] / 2dy
     
        # ∂h[n, j]/∂h[n, -1 + j] = μ/Power(dx,2) + u(n,j)/(2.*dx)
        jac[k, k_j_m] = μ / dx^2 + u[k] / 2dx
     
        # ∂h[n, j]/∂h[n, 1 + j] = μ/Power(dx,2) - u(n,j)/(2.*dx)
        jac[k, k_j_p] = μ / dx^2 - u[k] / 2dx
     
        # For boundary case {i -> n} looking at u[_, _]:
     
        # ∂h[n, j]/∂u[n, j] = -0.5*(-h(n,-1 + j) + h(n,1 + j))/dx
        jac[k, k + nx*ny] = -(-h[k_j_m] + h[k_j_p]) / 2dx
     
        # ∂h[n, j]/∂u[n, -1 + j] = h(n,j)/(2.*dx)
        jac[k, k_j_m + nx*ny] = h[k] / 2dx
     
        # ∂h[n, j]/∂u[n, 1 + j] = -0.5*h(n,j)/dx
        jac[k, k_j_p + nx*ny] = -h[k] / 2dx
     
        # For boundary case {i -> n} looking at v[_, _]:
     
        # ∂h[n, j]/∂v[-1 + n, j] = h(n,j)/(2.*dy)
        jac[k, k_i_m + 2nx*ny] = h[k] / 2dy
     
        # ∂h[n, j]/∂v[n, j] = h(n,j)/(2.*dy) - (-h(-1 + n,j) + h(n,j))/(2.*dy)
        jac[k, k + 2nx*ny] = h[k] / 2dy - (-h[k_i_m] + h[k]) / 2dy
     end

    for i in 2:ny-1
        k = flat_idx(i, 1, nx)
        k_i_m = flat_idx(i-1, 1, nx)
        k_i_p = flat_idx(i+1, 1, nx)
        k_j_p = flat_idx(i, 2, nx)
     
        # For boundary case {j -> 1} looking at h[_, _]:
     
        # ∂h[i, 1]/∂h[i, 1] = -(μ/Power(dx,2)) - (2*μ)/Power(dy,2) + u(i,1)/(2.*dx) - (u(i,1) + u(i,2))/(2.*dx) - (-v(-1 + i,1) + v(1 + i,1))/(2.*dy)
        jac[k, k] = -μ / dx^2 - 2μ / dy^2 + u[k] / 2dx - (u[k] + u[k_j_p]) / 2dx - (-v[k_i_m] + v[k_i_p]) / 2dy
     
        # ∂h[i, 1]/∂h[i, 2] = μ/Power(dx,2) - u(i,1)/(2.*dx)
        jac[k, k_j_p] = μ / dx^2 - u[k] / 2dx
     
        # ∂h[i, 1]/∂h[-1 + i, 1] = μ/Power(dy,2) + v(i,1)/(2.*dy)
        jac[k, k_i_m] = μ / dy^2 + v[k] / 2dy
     
        # ∂h[i, 1]/∂h[1 + i, 1] = μ/Power(dy,2) - v(i,1)/(2.*dy)
        jac[k, k_i_p] = μ / dy^2 - v[k] / 2dy
     
        # For boundary case {j -> 1} looking at u[_, _]:
     
        # ∂h[i, 1]/∂u[i, 1] = -0.5*h(i,1)/dx - (-h(i,1) + h(i,2))/(2.*dx)
        jac[k, k + nx*ny] = -h[k] / 2dx - (-h[k] + h[k_j_p]) / 2dx
     
        # ∂h[i, 1]/∂u[i, 2] = -0.5*h(i,1)/dx
        jac[k, k_j_p + nx*ny] = -h[k] / 2dx
     
        # For boundary case {j -> 1} looking at v[_, _]:
     
        # ∂h[i, 1]/∂v[i, 1] = -0.5*(-h(-1 + i,1) + h(1 + i,1))/dy
        jac[k, k + 2nx*ny] = -(-h[k_i_m] + h[k_i_p]) / 2dy
     
        # ∂h[i, 1]/∂v[-1 + i, 1] = h(i,1)/(2.*dy)
        jac[k, k_i_m + 2nx*ny] = h[k] / 2dy
     
        # ∂h[i, 1]/∂v[1 + i, 1] = -0.5*h(i,1)/dy
        jac[k, k_i_p + 2nx*ny] = -h[k] / 2dy
     end

    for i in 2:ny-1
        k = flat_idx(i, nx, nx)
        k_i_m = flat_idx(i-1, nx, nx)
        k_i_p = flat_idx(i+1, nx, nx)
        k_j_m = flat_idx(i, nx-1, nx)
     
        # For boundary case {j -> n} looking at h[_, _]:
     
        # ∂h[i, n]/∂h[i, -1 + n] = μ/Power(dx,2) + u(i,n)/(2.*dx)
        jac[k, k_j_m] = μ / dx^2 + u[k] / 2dx
     
        # ∂h[i, n]/∂h[i, n] = -(μ/Power(dx,2)) - (2*μ)/Power(dy,2) - (-u(i,-1 + n) - u(i,n))/(2.*dx) - u(i,n)/(2.*dx) - (-v(-1 + i,n) + v(1 + i,n))/(2.*dy)
        jac[k, k] = -μ / dx^2 - 2μ / dy^2 - (-u[k_j_m] - u[k]) / 2dx - u[k] / 2dx - (-v[k_i_m] + v[k_i_p]) / 2dy
     
        # ∂h[i, n]/∂h[-1 + i, n] = μ/Power(dy,2) + v(i,n)/(2.*dy)
        jac[k, k_i_m] = μ / dy^2 + v[k] / 2dy
     
        # ∂h[i, n]/∂h[1 + i, n] = μ/Power(dy,2) - v(i,n)/(2.*dy)
        jac[k, k_i_p] = μ / dy^2 - v[k] / 2dy
     
        # For boundary case {j -> n} looking at u[_, _]:
     
        # ∂h[i, n]/∂u[i, -1 + n] = h(i,n)/(2.*dx)
        jac[k, k_j_m + nx*ny] = h[k] / 2dx
     
        # ∂h[i, n]/∂u[i, n] = h(i,n)/(2.*dx) - (-h(i,-1 + n) + h(i,n))/(2.*dx)
        jac[k, k + nx*ny] = h[k] / 2dx - (-h[k_j_m] + h[k]) / 2dx
     
        # For boundary case {j -> n} looking at v[_, _]:
     
        # ∂h[i, n]/∂v[i, n] = -0.5*(-h(-1 + i,n) + h(1 + i,n))/dy
        jac[k, k + 2nx*ny] = -(-h[k_i_m] + h[k_i_p]) / 2dy
     
        # ∂h[i, n]/∂v[-1 + i, n] = h(i,n)/(2.*dy)
        jac[k, k_i_m + 2nx*ny] = h[k] / 2dy
     
        # ∂h[i, n]/∂v[1 + i, n] = -0.5*h(i,n)/dy
        jac[k, k_i_p + 2nx*ny] = -h[k] / 2dy
     end

    
    ### udot boundaries ###

    for j in 2:nx-1
        k = flat_idx(1, j, nx)
        k_i_p = flat_idx(2, j, nx)
        k_j_m = flat_idx(1, j-1, nx)
        k_j_p = flat_idx(1, j+1, nx)
     
        # For boundary case {i -> 1} looking at h[_, _]:
     
        # ∂u[1, j]/∂h[1, -1 + j] = g/(2.*dx)
        jac[k + nx*ny, k_j_m] = g / 2dx
     
        # ∂u[1, j]/∂h[1, 1 + j] = -0.5*g/dx
        jac[k + nx*ny, k_j_p] = -g / 2dx
     
        # For boundary case {i -> 1} looking at u[_, _]:
     
        # ∂u[1, j]/∂u[1, j] = (-2*ν)/Power(dx,2) - (-u(1,-1 + j) + u(1,1 + j))/(2.*dx) - v(1,j)/(2.*dy)
        jac[k + nx*ny, k + nx*ny] = -2ν / dx^2 - (-u[k_j_m] + u[k_j_p]) / 2dx - v[k] / 2dy
     
        # ∂u[1, j]/∂u[1, -1 + j] = ν/Power(dx,2) + u(1,j)/(2.*dx)
        jac[k + nx*ny, k_j_m + nx*ny] = ν / dx^2 + u[k] / 2dx
     
        # ∂u[1, j]/∂u[1, 1 + j] = ν/Power(dx,2) - u(1,j)/(2.*dx)
        jac[k + nx*ny, k_j_p + nx*ny] = ν / dx^2 - u[k] / 2dx
     
        # ∂u[1, j]/∂u[2, j] = -0.5*v(1,j)/dy
        jac[k + nx*ny, k_i_p + nx*ny] = -v[k] / 2dy
     
        # For boundary case {i -> 1} looking at v[_, _]:
     
        # ∂u[1, j]/∂v[1, j] = -0.5*(u(1,j) + u(2,j))/dy
        jac[k + nx*ny, k + 2nx*ny] = -(u[k] + u[k_i_p]) / 2dy
     end

    for j in 2:nx-1
        k = flat_idx(ny, j, nx)
        k_i_m = flat_idx(ny-1, j, nx)
        k_j_m = flat_idx(ny, j-1, nx)
        k_j_p = flat_idx(ny, j+1, nx)
     
        # For boundary case {i -> n} looking at h[_, _]:
     
        # ∂u[n, j]/∂h[n, -1 + j] = g/(2.*dx)
        jac[k + nx*ny, k_j_m] = g / 2dx
     
        # ∂u[n, j]/∂h[n, 1 + j] = -0.5*g/dx
        jac[k + nx*ny, k_j_p] = -g / 2dx
     
        # For boundary case {i -> n} looking at u[_, _]:
     
        # ∂u[n, j]/∂u[n, j] = (-2*ν)/Power(dx,2) - (-u(n,-1 + j) + u(n,1 + j))/(2.*dx) + v(n,j)/(2.*dy)
        jac[k + nx*ny, k + nx*ny] = -2ν / dx^2 - (-u[k_j_m] + u[k_j_p]) / 2dx + v[k] / 2dy
     
        # ∂u[n, j]/∂u[n, -1 + j] = ν/Power(dx,2) + u(n,j)/(2.*dx)
        jac[k + nx*ny, k_j_m + nx*ny] = ν / dx^2 + u[k] / 2dx
     
        # ∂u[n, j]/∂u[n, 1 + j] = ν/Power(dx,2) - u(n,j)/(2.*dx)
        jac[k + nx*ny, k_j_p + nx*ny] = ν / dx^2 - u[k] / 2dx
     
        # ∂u[n, j]/∂u[-1 + n, j] = v(n,j)/(2.*dy)
        jac[k + nx*ny, k_i_m + nx*ny] = v[k] / 2dy
     
        # For boundary case {i -> n} looking at v[_, _]:
     
        # ∂u[n, j]/∂v[n, j] = -0.5*(-u(-1 + n,j) - u(n,j))/dy
        jac[k + nx*ny, k + 2nx*ny] = -(-u[k_i_m] - u[k]) / 2dy
     end

    for i in 2:ny-1
        k = flat_idx(i, 1, nx)
        k_i_m = flat_idx(i-1, 1, nx)
        k_i_p = flat_idx(i+1, 1, nx)
        k_j_p = flat_idx(i, 2, nx)
     
        # For boundary case {j -> 1} looking at h[_, _]:
     
        # ∂u[i, 1]/∂h[i, 1] = g/(2.*dx)
        jac[k + nx*ny, k] = g / 2dx
     
        # ∂u[i, 1]/∂h[i, 2] = -0.5*g/dx
        jac[k + nx*ny, k_j_p] = -g / 2dx
     
        # For boundary case {j -> 1} looking at u[_, _]:
     
        # ∂u[i, 1]/∂u[i, 1] = (-3*ν)/Power(dx,2) - u(i,1)/(2.*dx) - (u(i,1) + u(i,2))/(2.*dx)
        jac[k + nx*ny, k + nx*ny] = -3ν / dx^2 - u[k] / 2dx - (u[k] + u[k_j_p]) / 2dx
     
        # ∂u[i, 1]/∂u[i, 2] = ν/Power(dx,2) - u(i,1)/(2.*dx)
        jac[k + nx*ny, k_j_p + nx*ny] = ν / dx^2 - u[k] / 2dx
     
        # ∂u[i, 1]/∂u[-1 + i, 1] = v(i,1)/(2.*dy)
        jac[k + nx*ny, k_i_m + nx*ny] = v[k] / 2dy
     
        # ∂u[i, 1]/∂u[1 + i, 1] = -0.5*v(i,1)/dy
        jac[k + nx*ny, k_i_p + nx*ny] = -v[k] / 2dy
     
        # For boundary case {j -> 1} looking at v[_, _]:
     
        # ∂u[i, 1]/∂v[i, 1] = -0.5*(-u(-1 + i,1) + u(1 + i,1))/dy
        jac[k + nx*ny, k + 2nx*ny] = -(-u[k_i_m] + u[k_i_p]) / 2dy
     end

    for i in 2:ny-1
        k = flat_idx(i, nx, nx)
        k_i_m = flat_idx(i-1, nx, nx)
        k_i_p = flat_idx(i+1, nx, nx)
        k_j_m = flat_idx(i, nx-1, nx)
     
        # For boundary case {j -> n} looking at h[_, _]:
     
        # ∂u[i, n]/∂h[i, -1 + n] = g/(2.*dx)
        jac[k + nx*ny, k_j_m] = g / 2dx
     
        # ∂u[i, n]/∂h[i, n] = -0.5*g/dx
        jac[k + nx*ny, k] = -g / 2dx
     
        # For boundary case {j -> n} looking at u[_, _]:
     
        # ∂u[i, n]/∂u[i, -1 + n] = ν/Power(dx,2) + u(i,n)/(2.*dx)
        jac[k + nx*ny, k_j_m + nx*ny] = ν / dx^2 + u[k] / 2dx
     
        # ∂u[i, n]/∂u[i, n] = (-3*ν)/Power(dx,2) - (-u(i,-1 + n) - u(i,n))/(2.*dx) + u(i,n)/(2.*dx)
        jac[k + nx*ny, k + nx*ny] = -3ν / dx^2 - (-u[k_j_m] - u[k]) / 2dx + u[k] / 2dx
     
        # ∂u[i, n]/∂u[-1 + i, n] = v(i,n)/(2.*dy)
        jac[k + nx*ny, k_i_m + nx*ny] = v[k] / 2dy
     
        # ∂u[i, n]/∂u[1 + i, n] = -0.5*v(i,n)/dy
        jac[k + nx*ny, k_i_p + nx*ny] = -v[k] / 2dy
     
        # For boundary case {j -> n} looking at v[_, _]:
     
        # ∂u[i, n]/∂v[i, n] = -0.5*(-u(-1 + i,n) + u(1 + i,n))/dy
        jac[k + nx*ny, k + 2nx*ny] = -(-u[k_i_m] + u[k_i_p]) / 2dy
     end


    ### vdot boundaries ###

    for j in 2:nx-1
        k = flat_idx(1, j, nx)
        k_i_p = flat_idx(2, j, nx)
        k_j_m = flat_idx(1, j-1, nx)
        k_j_p = flat_idx(1, j+1, nx)
     
        # For boundary case {i -> 1} looking at h[_, _]:
     
        # ∂v[1, j]/∂h[1, j] = g/(2.*dy)
        jac[k + 2nx*ny, k] = g / 2dy
     
        # ∂v[1, j]/∂h[2, j] = -0.5*g/dy
        jac[k + 2nx*ny, k_i_p] = -g / 2dy
     
        # For boundary case {i -> 1} looking at u[_, _]:
     
        # ∂v[1, j]/∂u[1, j] = -0.5*(-v(1,-1 + j) + v(1,1 + j))/dx
        jac[k + 2nx*ny, k + nx*ny] = -(-v[k_j_m] + v[k_j_p]) / 2dx
     
        # For boundary case {i -> 1} looking at v[_, _]:
     
        # ∂v[1, j]/∂v[1, -1 + j] = u(1,j)/(2.*dx)
        jac[k + 2nx*ny, k_j_m + 2nx*ny] = u[k] / 2dx
     
        # ∂v[1, j]/∂v[1, 1 + j] = -0.5*u(1,j)/dx
        jac[k + 2nx*ny, k_j_p + 2nx*ny] = -u[k] / 2dx
     
        # ∂v[1, j]/∂v[1, j] = (-3*ν)/Power(dy,2) - v(1,j)/(2.*dy) - (v(1,j) + v(2,j))/(2.*dy)
        jac[k + 2nx*ny, k + 2nx*ny] = -3ν / dy^2 - v[k] / 2dy - (v[k] + v[k_i_p]) / 2dy
     
        # ∂v[1, j]/∂v[2, j] = ν/Power(dy,2) - v(1,j)/(2.*dy)
        jac[k + 2nx*ny, k_i_p + 2nx*ny] = ν / dy^2 - v[k] / 2dy
     end

    for j in 2:nx-1
        k = flat_idx(ny, j, nx)
        k_i_m = flat_idx(ny-1, j, nx)
        k_j_m = flat_idx(ny, j-1, nx)
        k_j_p = flat_idx(ny, j+1, nx)
     
        # For boundary case {i -> n} looking at h[_, _]:
     
        # ∂v[n, j]/∂h[-1 + n, j] = g/(2.*dy)
        jac[k + 2nx*ny, k_i_m] = g / 2dy
     
        # ∂v[n, j]/∂h[n, j] = -0.5*g/dy
        jac[k + 2nx*ny, k] = -g / 2dy
     
        # For boundary case {i -> n} looking at u[_, _]:
     
        # ∂v[n, j]/∂u[n, j] = -0.5*(-v(n,-1 + j) + v(n,1 + j))/dx
        jac[k + 2nx*ny, k + nx*ny] = -(-v[k_j_m] + v[k_j_p]) / 2dx
     
        # For boundary case {i -> n} looking at v[_, _]:
     
        # ∂v[n, j]/∂v[-1 + n, j] = ν/Power(dy,2) + v(n,j)/(2.*dy)
        jac[k + 2nx*ny, k_i_m + 2nx*ny] = ν / dy^2 + v[k] / 2dy
     
        # ∂v[n, j]/∂v[n, j] = (-3*ν)/Power(dy,2) - (-v(-1 + n,j) - v(n,j))/(2.*dy) + v(n,j)/(2.*dy)
        jac[k + 2nx*ny, k + 2nx*ny] = -3ν / dy^2 - (-v[k_i_m] - v[k]) / 2dy + v[k] / 2dy
     
        # ∂v[n, j]/∂v[n, -1 + j] = u(n,j)/(2.*dx)
        jac[k + 2nx*ny, k_j_m + 2nx*ny] = u[k] / 2dx
     
        # ∂v[n, j]/∂v[n, 1 + j] = -0.5*u(n,j)/dx
        jac[k + 2nx*ny, k_j_p + 2nx*ny] = -u[k] / 2dx
     end

    for i in 2:ny-1
        k = flat_idx(i, 1, nx)
        k_i_m = flat_idx(i-1, 1, nx)
        k_i_p = flat_idx(i+1, 1, nx)
        k_j_p = flat_idx(i, 2, nx)
     
        # For boundary case {j -> 1} looking at h[_, _]:
     
        # ∂v[i, 1]/∂h[-1 + i, 1] = g/(2.*dy)
        jac[k + 2nx*ny, k_i_m] = g / 2dy
     
        # ∂v[i, 1]/∂h[1 + i, 1] = -0.5*g/dy
        jac[k + 2nx*ny, k_i_p] = -g / 2dy
     
        # For boundary case {j -> 1} looking at u[_, _]:
     
        # ∂v[i, 1]/∂u[i, 1] = -0.5*(v(i,1) + v(i,2))/dx
        jac[k + 2nx*ny, k + nx*ny] = -(v[k] + v[k_j_p]) / 2dx
     
        # For boundary case {j -> 1} looking at v[_, _]:
     
        # ∂v[i, 1]/∂v[i, 1] = (-2*ν)/Power(dy,2) - u(i,1)/(2.*dx) - (-v(-1 + i,1) + v(1 + i,1))/(2.*dy)
        jac[k + 2nx*ny, k + 2nx*ny] = -2ν / dy^2 - u[k] / 2dx - (-v[k_i_m] + v[k_i_p]) / 2dy
     
        # ∂v[i, 1]/∂v[i, 2] = -0.5*u(i,1)/dx
        jac[k + 2nx*ny, k_j_p + 2nx*ny] = -u[k] / 2dx
     
        # ∂v[i, 1]/∂v[-1 + i, 1] = ν/Power(dy,2) + v(i,1)/(2.*dy)
        jac[k + 2nx*ny, k_i_m + 2nx*ny] = ν / dy^2 + v[k] / 2dy
     
        # ∂v[i, 1]/∂v[1 + i, 1] = ν/Power(dy,2) - v(i,1)/(2.*dy)
        jac[k + 2nx*ny, k_i_p + 2nx*ny] = ν / dy^2 - v[k] / 2dy
     end

    for i in 2:ny-1
        k = flat_idx(i, nx, nx)
        k_i_m = flat_idx(i-1, nx, nx)
        k_i_p = flat_idx(i+1, nx, nx)
        k_j_m = flat_idx(i, nx-1, nx)
     
        # For boundary case {j -> n} looking at h[_, _]:
     
        # ∂v[i, n]/∂h[-1 + i, n] = g/(2.*dy)
        jac[k + 2nx*ny, k_i_m] = g / 2dy
     
        # ∂v[i, n]/∂h[1 + i, n] = -0.5*g/dy
        jac[k + 2nx*ny, k_i_p] = -g / 2dy
     
        # For boundary case {j -> n} looking at u[_, _]:
     
        # ∂v[i, n]/∂u[i, n] = -0.5*(-v(i,-1 + n) - v(i,n))/dx
        jac[k + 2nx*ny, k + nx*ny] = -(-v[k_j_m] - v[k]) / 2dx
     
        # For boundary case {j -> n} looking at v[_, _]:
     
        # ∂v[i, n]/∂v[i, -1 + n] = u(i,n)/(2.*dx)
        jac[k + 2nx*ny, k_j_m + 2nx*ny] = u[k] / 2dx
     
        # ∂v[i, n]/∂v[i, n] = (-2*ν)/Power(dy,2) + u(i,n)/(2.*dx) - (-v(-1 + i,n) + v(1 + i,n))/(2.*dy)
        jac[k + 2nx*ny, k + 2nx*ny] = -2ν / dy^2 + u[k] / 2dx - (-v[k_i_m] + v[k_i_p]) / 2dy
     
        # ∂v[i, n]/∂v[-1 + i, n] = ν/Power(dy,2) + v(i,n)/(2.*dy)
        jac[k + 2nx*ny, k_i_m + 2nx*ny] = ν / dy^2 + v[k] / 2dy
     
        # ∂v[i, n]/∂v[1 + i, n] = ν/Power(dy,2) - v(i,n)/(2.*dy)
        jac[k + 2nx*ny, k_i_p + 2nx*ny] = ν / dy^2 - v[k] / 2dy
     end
end


function jacobian_2D_corner_terms!(jac, h, u, v, prob::ShallowWaterProblem2D)
    nx = prob.nx
    ny = prob.ny
    dx = prob.dx
    dy = prob.dy
    μ = prob.μ
    ν = prob.ν
    g = 9.81

    ### hdot boundaries ###

    # Corner (1,1)
    k = flat_idx(1, 1, nx)
    k_i_p = flat_idx(2, 1, nx)
    k_j_p = flat_idx(1, 2, nx)

    # For boundary case {i -> 1, j -> 1} looking at h[_, _]:

    # ∂h[1, 1]/∂h[1, 1] = -(μ/Power(dx,2)) - μ/Power(dy,2) + u(1,1)/(2.*dx) - (u(1,1) + u(1,2))/(2.*dx) + v(1,1)/(2.*dy) - (v(1,1) + v(2,1))/(2.*dy)
    jac[k, k] = -μ / dx^2 - μ / dy^2 + u[k] / 2dx - (u[k] + u[k_j_p]) / 2dx + v[k] / 2dy - (v[k] + v[k_i_p]) / 2dy

    # ∂h[1, 1]/∂h[1, 2] = μ/Power(dx,2) - u(1,1)/(2.*dx)
    jac[k, k_j_p] = μ / dx^2 - u[k] / 2dx

    # ∂h[1, 1]/∂h[2, 1] = μ/Power(dy,2) - v(1,1)/(2.*dy)
    jac[k, k_i_p] = μ / dy^2 - v[k] / 2dy

    # For boundary case {i -> 1, j -> 1} looking at u[_, _]:

    # ∂h[1, 1]/∂u[1, 1] = -0.5*h(1,1)/dx - (-h(1,1) + h(1,2))/(2.*dx)
    jac[k, k + nx*ny] = -h[k] / 2dx - (-h[k] + h[k_j_p]) / 2dx

    # ∂h[1, 1]/∂u[1, 2] = -0.5*h(1,1)/dx
    jac[k, k_j_p + nx*ny] = -h[k] / 2dx

    # For boundary case {i -> 1, j -> 1} looking at v[_, _]:

    # ∂h[1, 1]/∂v[1, 1] = -0.5*h(1,1)/dy - (-h(1,1) + h(2,1))/(2.*dy)
    jac[k, k + 2nx*ny] = -h[k] / 2dy - (-h[k] + h[k_i_p]) / 2dy

    # ∂h[1, 1]/∂v[2, 1] = -0.5*h(1,1)/dy
    jac[k, k_i_p + 2nx*ny] = -h[k] / 2dy


    # Corner (1,n)
    k = flat_idx(1, nx, nx)
    k_i_p = flat_idx(2, nx, nx)
    k_j_m = flat_idx(1, nx-1, nx)

    # For boundary case {i -> 1, j -> n} looking at h[_, _]:

    # ∂h[1, n]/∂h[1, -1 + n] = μ/Power(dx,2) + u(1,n)/(2.*dx)
    jac[k, k_j_m] = μ / dx^2 + u[k] / 2dx

    # ∂h[1, n]/∂h[1, n] = -(μ/Power(dx,2)) - μ/Power(dy,2) - (-u(1,-1 + n) - u(1,n))/(2.*dx) - u(1,n)/(2.*dx) + v(1,n)/(2.*dy) - (v(1,n) + v(2,n))/(2.*dy)
    jac[k, k] = -μ / dx^2 - μ / dy^2 - (-u[k_j_m] - u[k]) / 2dx - u[k] / 2dx + v[k] / 2dy - (v[k] + v[k_i_p]) / 2dy

    # ∂h[1, n]/∂h[2, n] = μ/Power(dy,2) - v(1,n)/(2.*dy)
    jac[k, k_i_p] = μ / dy^2 - v[k] / 2dy

    # For boundary case {i -> 1, j -> n} looking at u[_, _]:

    # ∂h[1, n]/∂u[1, -1 + n] = h(1,n)/(2.*dx)
    jac[k, k_j_m + nx*ny] = h[k] / 2dx

    # ∂h[1, n]/∂u[1, n] = h(1,n)/(2.*dx) - (-h(1,-1 + n) + h(1,n))/(2.*dx)
    jac[k, k + nx*ny] = h[k] / 2dx - (-h[k_j_m] + h[k]) / 2dx

    # For boundary case {i -> 1, j -> n} looking at v[_, _]:

    # ∂h[1, n]/∂v[1, n] = -0.5*h(1,n)/dy - (-h(1,n) + h(2,n))/(2.*dy)
    jac[k, k + 2nx*ny] = -h[k] / 2dy - (-h[k] + h[k_i_p]) / 2dy

    # ∂h[1, n]/∂v[2, n] = -0.5*h(1,n)/dy
    jac[k, k_i_p + 2nx*ny] = -h[k] / 2dy


    # Corner (n,1)
    k = flat_idx(ny, 1, nx)
    k_i_m = flat_idx(ny-1, 1, nx)
    k_j_p = flat_idx(ny, 2, nx)

    # For boundary case {i -> n, j -> 1} looking at h[_, _]:

    # ∂h[n, 1]/∂h[-1 + n, 1] = μ/Power(dy,2) + v(n,1)/(2.*dy)
    jac[k, k_i_m] = μ / dy^2 + v[k] / 2dy

    # ∂h[n, 1]/∂h[n, 1] = -(μ/Power(dx,2)) - μ/Power(dy,2) + u(n,1)/(2.*dx) - (u(n,1) + u(n,2))/(2.*dx) - (-v(-1 + n,1) - v(n,1))/(2.*dy) - v(n,1)/(2.*dy)
    jac[k, k] = -μ / dx^2 - μ / dy^2 + u[k] / 2dx - (u[k] + u[k_j_p]) / 2dx - (-v[k_i_m] - v[k]) / 2dy - v[k] / 2dy

    # ∂h[n, 1]/∂h[n, 2] = μ/Power(dx,2) - u(n,1)/(2.*dx)
    jac[k, k_j_p] = μ / dx^2 - u[k] / 2dx

    # For boundary case {i -> n, j -> 1} looking at u[_, _]:

    # ∂h[n, 1]/∂u[n, 1] = -0.5*h(n,1)/dx - (-h(n,1) + h(n,2))/(2.*dx)
    jac[k, k + nx*ny] = -h[k] / 2dx - (-h[k] + h[k_j_p]) / 2dx

    # ∂h[n, 1]/∂u[n, 2] = -0.5*h(n,1)/dx
    jac[k, k_j_p + nx*ny] = -h[k] / 2dx

    # For boundary case {i -> n, j -> 1} looking at v[_, _]:

    # ∂h[n, 1]/∂v[-1 + n, 1] = h(n,1)/(2.*dy)
    jac[k, k_i_m + 2nx*ny] = h[k] / 2dy

    # ∂h[n, 1]/∂v[n, 1] = h(n,1)/(2.*dy) - (-h(-1 + n,1) + h(n,1))/(2.*dy)
    jac[k, k + 2nx*ny] = h[k] / 2dy - (-h[k_i_m] + h[k]) / 2dy


    # Corner (n,n)
    k = flat_idx(ny, nx, nx)
    k_i_m = flat_idx(ny-1, nx, nx)
    k_j_m = flat_idx(ny, nx-1, nx)

    # For boundary case {i -> n, j -> n} looking at h[_, _]:

    # ∂h[n, n]/∂h[-1 + n, n] = μ/Power(dy,2) + v(n,n)/(2.*dy)
    jac[k, k_i_m] = μ / dy^2 + v[k] / 2dy

    # ∂h[n, n]/∂h[n, n] = -(μ/Power(dx,2)) - μ/Power(dy,2) - (-u(n,-1 + n) - u(n,n))/(2.*dx) - u(n,n)/(2.*dx) - (-v(-1 + n,n) - v(n,n))/(2.*dy) - v(n,n)/(2.*dy)
    jac[k, k] = -μ / dx^2 - μ / dy^2 - (-u[k_j_m] - u[k]) / 2dx - u[k] / 2dx - (-v[k_i_m] - v[k]) / 2dy - v[k] / 2dy

    # ∂h[n, n]/∂h[n, -1 + n] = μ/Power(dx,2) + u(n,n)/(2.*dx)
    jac[k, k_j_m] = μ / dx^2 + u[k] / 2dx

    # For boundary case {i -> n, j -> n} looking at u[_, _]:

    # ∂h[n, n]/∂u[n, -1 + n] = h(n,n)/(2.*dx)
    jac[k, k_j_m + nx*ny] = h[k] / 2dx

    # ∂h[n, n]/∂u[n, n] = h(n,n)/(2.*dx) - (-h(n,-1 + n) + h(n,n))/(2.*dx)
    jac[k, k + nx*ny] = h[k] / 2dx - (-h[k_j_m] + h[k]) / 2dx

    # For boundary case {i -> n, j -> n} looking at v[_, _]:

    # ∂h[n, n]/∂v[-1 + n, n] = h(n,n)/(2.*dy)
    jac[k, k_i_m + 2nx*ny] = h[k] / 2dy

    # ∂h[n, n]/∂v[n, n] = h(n,n)/(2.*dy) - (-h(-1 + n,n) + h(n,n))/(2.*dy)
    jac[k, k + 2nx*ny] = h[k] / 2dy - (-h[k_i_m] + h[k]) / 2dy


    ### udot boundaries ###

    # Corner (1,1) for u equation
    k = flat_idx(1, 1, nx)
    k_i_p = flat_idx(2, 1, nx)
    k_j_p = flat_idx(1, 2, nx)

    # For boundary case {i -> 1, j -> 1} looking at h[_, _]:

    # ∂u[1, 1]/∂h[1, 1] = g/(2.*dx)
    jac[k + nx*ny, k] = g / 2dx

    # ∂u[1, 1]/∂h[1, 2] = -0.5*g/dx
    jac[k + nx*ny, k_j_p] = -g / 2dx

    # For boundary case {i -> 1, j -> 1} looking at u[_, _]:

    # ∂u[1, 1]/∂u[1, 1] = (-3*ν)/Power(dx,2) - u(1,1)/(2.*dx) - (u(1,1) + u(1,2))/(2.*dx) - v(1,1)/(2.*dy)
    jac[k + nx*ny, k + nx*ny] = -3ν / dx^2 - u[k] / 2dx - (u[k] + u[k_j_p]) / 2dx - v[k] / 2dy

    # ∂u[1, 1]/∂u[1, 2] = ν/Power(dx,2) - u(1,1)/(2.*dx)
    jac[k + nx*ny, k_j_p + nx*ny] = ν / dx^2 - u[k] / 2dx

    # ∂u[1, 1]/∂u[2, 1] = -0.5*v(1,1)/dy
    jac[k + nx*ny, k_i_p + nx*ny] = -v[k] / 2dy

    # For boundary case {i -> 1, j -> 1} looking at v[_, _]:

    # ∂u[1, 1]/∂v[1, 1] = -0.5*(u(1,1) + u(2,1))/dy
    jac[k + nx*ny, k + 2nx*ny] = -(u[k] + u[k_i_p]) / 2dy

    # Corner (1,n) for u equation
    k = flat_idx(1, nx, nx)
    k_i_p = flat_idx(2, nx, nx)
    k_j_m = flat_idx(1, nx-1, nx)

    # For boundary case {i -> 1, j -> n} looking at h[_, _]:

    # ∂u[1, n]/∂h[1, -1 + n] = g/(2.*dx)
    jac[k + nx*ny, k_j_m] = g / 2dx

    # ∂u[1, n]/∂h[1, n] = -0.5*g/dx
    jac[k + nx*ny, k] = -g / 2dx

    # For boundary case {i -> 1, j -> n} looking at u[_, _]:

    # ∂u[1, n]/∂u[1, -1 + n] = ν/Power(dx,2) + u(1,n)/(2.*dx)
    jac[k + nx*ny, k_j_m + nx*ny] = ν / dx^2 + u[k] / 2dx

    # ∂u[1, n]/∂u[1, n] = (-3*ν)/Power(dx,2) - (-u(1,-1 + n) - u(1,n))/(2.*dx) + u(1,n)/(2.*dx) - v(1,n)/(2.*dy)
    jac[k + nx*ny, k + nx*ny] = -3ν / dx^2 - (-u[k_j_m] - u[k]) / 2dx + u[k] / 2dx - v[k] / 2dy

    # ∂u[1, n]/∂u[2, n] = -0.5*v(1,n)/dy
    jac[k + nx*ny, k_i_p + nx*ny] = -v[k] / 2dy

    # For boundary case {i -> 1, j -> n} looking at v[_, _]:

    # ∂u[1, n]/∂v[1, n] = -0.5*(u(1,n) + u(2,n))/dy
    jac[k + nx*ny, k + 2nx*ny] = -(u[k] + u[k_i_p]) / 2dy

    # Corner (n,1) for u equation
    k = flat_idx(ny, 1, nx)
    k_i_m = flat_idx(ny-1, 1, nx)
    k_j_p = flat_idx(ny, 2, nx)

    # For boundary case {i -> n, j -> 1} looking at h[_, _]:

    # ∂u[n, 1]/∂h[n, 1] = g/(2.*dx)
    jac[k + nx*ny, k] = g / 2dx

    # ∂u[n, 1]/∂h[n, 2] = -0.5*g/dx
    jac[k + nx*ny, k_j_p] = -g / 2dx

    # For boundary case {i -> n, j -> 1} looking at u[_, _]:

    # ∂u[n, 1]/∂u[n, 1] = (-3*ν)/Power(dx,2) - u(n,1)/(2.*dx) - (u(n,1) + u(n,2))/(2.*dx) + v(n,1)/(2.*dy)
    jac[k + nx*ny, k + nx*ny] = -3ν / dx^2 - u[k] / 2dx - (u[k] + u[k_j_p]) / 2dx + v[k] / 2dy

    # ∂u[n, 1]/∂u[n, 2] = ν/Power(dx,2) - u(n,1)/(2.*dx)
    jac[k + nx*ny, k_j_p + nx*ny] = ν / dx^2 - u[k] / 2dx

    # ∂u[n, 1]/∂u[-1 + n, 1] = v(n,1)/(2.*dy)
    jac[k + nx*ny, k_i_m + nx*ny] = v[k] / 2dy

    # For boundary case {i -> n, j -> 1} looking at v[_, _]:

    # ∂u[n, 1]/∂v[n, 1] = -0.5*(-u(-1 + n,1) - u(n,1))/dy
    jac[k + nx*ny, k + 2nx*ny] = -(-u[k_i_m] - u[k]) / 2dy

    # Corner (n,n) for u equation
    k = flat_idx(ny, nx, nx)
    k_i_m = flat_idx(ny-1, nx, nx)
    k_j_m = flat_idx(ny, nx-1, nx)

    # For boundary case {i -> n, j -> n} looking at h[_, _]:

    # ∂u[n, n]/∂h[n, -1 + n] = g/(2.*dx)
    jac[k + nx*ny, k_j_m] = g / 2dx

    # ∂u[n, n]/∂h[n, n] = -0.5*g/dx
    jac[k + nx*ny, k] = -g / 2dx

    # For boundary case {i -> n, j -> n} looking at u[_, _]:

    # ∂u[n, n]/∂u[n, -1 + n] = ν/Power(dx,2) + u(n,n)/(2.*dx)
    jac[k + nx*ny, k_j_m + nx*ny] = ν / dx^2 + u[k] / 2dx

    # ∂u[n, n]/∂u[n, n] = (-3*ν)/Power(dx,2) - (-u(n,-1 + n) - u(n,n))/(2.*dx) + u(n,n)/(2.*dx) + v(n,n)/(2.*dy)
    jac[k + nx*ny, k + nx*ny] = -3ν / dx^2 - (-u[k_j_m] - u[k]) / 2dx + u[k] / 2dx + v[k] / 2dy

    # ∂u[n, n]/∂u[-1 + n, n] = v(n,n)/(2.*dy)
    jac[k + nx*ny, k_i_m + nx*ny] = v[k] / 2dy

    # For boundary case {i -> n, j -> n} looking at v[_, _]:

    # ∂u[n, n]/∂v[n, n] = -0.5*(-u(-1 + n,n) - u(n,n))/dy
    jac[k + nx*ny, k + 2nx*ny] = -(-u[k_i_m] - u[k]) / 2dy


    ### vdot boundaries ###

    # Corner (1,1) for v equation
    k = flat_idx(1, 1, nx)
    k_i_p = flat_idx(2, 1, nx)
    k_j_p = flat_idx(1, 2, nx)

    # For boundary case {i -> 1, j -> 1} looking at h[_, _]:

    # ∂v[1, 1]/∂h[1, 1] = g/(2.*dy)
    jac[k + 2nx*ny, k] = g / 2dy

    # ∂v[1, 1]/∂h[2, 1] = -0.5*g/dy
    jac[k + 2nx*ny, k_i_p] = -g / 2dy

    # For boundary case {i -> 1, j -> 1} looking at u[_, _]:

    # ∂v[1, 1]/∂u[1, 1] = -0.5*(v(1,1) + v(1,2))/dx
    jac[k + 2nx*ny, k + nx*ny] = -(v[k] + v[k_j_p]) / 2dx

    # For boundary case {i -> 1, j -> 1} looking at v[_, _]:

    # ∂v[1, 1]/∂v[1, 1] = (-3*ν)/Power(dy,2) - u(1,1)/(2.*dx) - v(1,1)/(2.*dy) - (v(1,1) + v(2,1))/(2.*dy)
    jac[k + 2nx*ny, k + 2nx*ny] = -3ν / dy^2 - u[k] / 2dx - v[k] / 2dy - (v[k] + v[k_i_p]) / 2dy

    # ∂v[1, 1]/∂v[1, 2] = -0.5*u(1,1)/dx
    jac[k + 2nx*ny, k_j_p + 2nx*ny] = -u[k] / 2dx

    # ∂v[1, 1]/∂v[2, 1] = ν/Power(dy,2) - v(1,1)/(2.*dy)
    jac[k + 2nx*ny, k_i_p + 2nx*ny] = ν / dy^2 - v[k] / 2dy

    # Corner (1,n) for v equation
    k = flat_idx(1, nx, nx)
    k_i_p = flat_idx(2, nx, nx)
    k_j_m = flat_idx(1, nx-1, nx)

    # For boundary case {i -> 1, j -> n} looking at h[_, _]:

    # ∂v[1, n]/∂h[1, n] = g/(2.*dy)
    jac[k + 2nx*ny, k] = g / 2dy

    # ∂v[1, n]/∂h[2, n] = -0.5*g/dy
    jac[k + 2nx*ny, k_i_p] = -g / 2dy

    # For boundary case {i -> 1, j -> n} looking at u[_, _]:

    # ∂v[1, n]/∂u[1, n] = -0.5*(-v(1,-1 + n) - v(1,n))/dx
    jac[k + 2nx*ny, k + nx*ny] = -(-v[k_j_m] - v[k]) / 2dx

    # For boundary case {i -> 1, j -> n} looking at v[_, _]:

    # ∂v[1, n]/∂v[1, -1 + n] = u(1,n)/(2.*dx)
    jac[k + 2nx*ny, k_j_m + 2nx*ny] = u[k] / 2dx

    # ∂v[1, n]/∂v[1, n] = (-3*ν)/Power(dy,2) + u(1,n)/(2.*dx) - v(1,n)/(2.*dy) - (v(1,n) + v(2,n))/(2.*dy)
    jac[k + 2nx*ny, k + 2nx*ny] = -3ν / dy^2 + u[k] / 2dx - v[k] / 2dy - (v[k] + v[k_i_p]) / 2dy

    # ∂v[1, n]/∂v[2, n] = ν/Power(dy,2) - v(1,n)/(2.*dy)
    jac[k + 2nx*ny, k_i_p + 2nx*ny] = ν / dy^2 - v[k] / 2dy

    # Corner (n,1) for v equation
    k = flat_idx(ny, 1, nx)
    k_i_m = flat_idx(ny-1, 1, nx)
    k_j_p = flat_idx(ny, 2, nx)

    # For boundary case {i -> n, j -> 1} looking at h[_, _]:

    # ∂v[n, 1]/∂h[-1 + n, 1] = g/(2.*dy)
    jac[k + 2nx*ny, k_i_m] = g / 2dy

    # ∂v[n, 1]/∂h[n, 1] = -0.5*g/dy
    jac[k + 2nx*ny, k] = -g / 2dy

    # For boundary case {i -> n, j -> 1} looking at u[_, _]:

    # ∂v[n, 1]/∂u[n, 1] = -0.5*(v(n,1) + v(n,2))/dx
    jac[k + 2nx*ny, k + nx*ny] = -(v[k] + v[k_j_p]) / 2dx

    # For boundary case {i -> n, j -> 1} looking at v[_, _]:

    # ∂v[n, 1]/∂v[-1 + n, 1] = ν/Power(dy,2) + v(n,1)/(2.*dy)
    jac[k + 2nx*ny, k_i_m + 2nx*ny] = ν / dy^2 + v[k] / 2dy

    # ∂v[n, 1]/∂v[n, 1] = (-3*ν)/Power(dy,2) - u(n,1)/(2.*dx) - (-v(-1 + n,1) - v(n,1))/(2.*dy) + v(n,1)/(2.*dy)
    jac[k + 2nx*ny, k + 2nx*ny] = -3ν / dy^2 - u[k] / 2dx - (-v[k_i_m] - v[k]) / 2dy + v[k] / 2dy

    # ∂v[n, 1]/∂v[n, 2] = -0.5*u(n,1)/dx
    jac[k + 2nx*ny, k_j_p + 2nx*ny] = -u[k] / 2dx

    # Corner (n,n) for v equation
    k = flat_idx(ny, nx, nx)
    k_i_m = flat_idx(ny-1, nx, nx)
    k_j_m = flat_idx(ny, nx-1, nx)

    # For boundary case {i -> n, j -> n} looking at h[_, _]:

    # ∂v[n, n]/∂h[-1 + n, n] = g/(2.*dy)
    jac[k + 2nx*ny, k_i_m] = g / 2dy

    # ∂v[n, n]/∂h[n, n] = -0.5*g/dy
    jac[k + 2nx*ny, k] = -g / 2dy

    # For boundary case {i -> n, j -> n} looking at u[_, _]:

    # ∂v[n, n]/∂u[n, n] = -0.5*(-v(n,-1 + n) - v(n,n))/dx
    jac[k + 2nx*ny, k + nx*ny] = -(-v[k_j_m] - v[k]) / 2dx

    # For boundary case {i -> n, j -> n} looking at v[_, _]:

    # ∂v[n, n]/∂v[-1 + n, n] = ν/Power(dy,2) + v(n,n)/(2.*dy)
    jac[k + 2nx*ny, k_i_m + 2nx*ny] = ν / dy^2 + v[k] / 2dy

    # ∂v[n, n]/∂v[n, n] = (-3*ν)/Power(dy,2) + u(n,n)/(2.*dx) - (-v(-1 + n,n) - v(n,n))/(2.*dy) + v(n,n)/(2.*dy)
    jac[k + 2nx*ny, k + 2nx*ny] = -3ν / dy^2 + u[k] / 2dx - (-v[k_i_m] - v[k]) / 2dy + v[k] / 2dy

    # ∂v[n, n]/∂v[n, -1 + n] = u(n,n)/(2.*dx)
    jac[k + 2nx*ny, k_j_m + 2nx*ny] = u[k] / 2dx

end
