using LinearAlgebra
using Trapz


function jacobian!(j, h, u, prob::ShallowWaterProblem)
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
    return DiscreteAcceleration(new_values, acc.times)
end


function objective(sol, target)
    nx = length(target)
    h_final = sol.u[end][1:nx]
    return sum((h_final .- target).^2) / 2
end


function adjoint_gradient(prob, acc)
    # Solve forward in time
    sol = solve_forward(prob, acc)
    
    # Solve adjoint equation
    sol_adj = solve_adjoint(sol, prob, acc)
    
    # Compute gradient
    return compute_control_gradient(sol_adj, acc.times, prob.nx)
end


function optimize(prob::ShallowWaterProblem)
    max_time = prob.tspan[2]
    acc_points = 50
    # acc = DiscreteAcceleration(0.1 * randn(acc_points), collect(LinRange(0, max_time, acc_points + 1)))
    acc = DiscreteAcceleration(zeros(acc_points), collect(LinRange(0, max_time, acc_points + 1)))
    
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
