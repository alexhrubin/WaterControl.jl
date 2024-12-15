using WaterControl
using Test


# compute gradient from multiple forward solves.
function gradient(prob::ShallowWaterProblem1D, acc::DiscreteAcceleration1D)
    objectives_plus = []
    s = 0.01
    for idx in 1:length(acc.values)
        a = DiscreteAcceleration(collect(acc.values), collect(LinRange(0, 10, length(acc.times))))
        a[idx] += s
        sol = solve_forward(prob, a)
        obj = WaterControl.objective(sol, prob.target)
        push!(objectives_plus, obj)
    end
    
    objectives_minus = []
    for idx in 1:length(acc.values)
        a = DiscreteAcceleration(collect(acc.values), collect(LinRange(0, 10, length(acc.times))))
        a[idx] += -s
        sol = solve_forward(prob, a)
        obj = WaterControl.objective(sol, prob.target)
        push!(objectives_minus, obj)
    end

    return (objectives_plus - objectives_minus) ./ 2s
end


function gradient(prob::ShallowWaterProblem2D, acc::DiscreteAcceleration2D)
    s = 0.001
    n_x = length(acc.values_x)
    n_y = length(acc.values_y)
    gradients = zeros(n_x + n_y)
    
    for idx in 1:(n_x + n_y)
        a_plus = DiscreteAcceleration2D(
            collect(acc.values_x),
            collect(acc.values_y),
            collect(LinRange(0, 10, length(acc.times)))
        )
        a_minus = DiscreteAcceleration2D(
            collect(acc.values_x),
            collect(acc.values_y),
            collect(LinRange(0, 10, length(acc.times)))
        )
        
        if idx <= n_x
            a_plus.values_x[idx] += s
            a_minus.values_x[idx] -= s
        else
            a_plus.values_y[idx - n_x] += s
            a_minus.values_y[idx - n_x] -= s
        end
        
        obj_plus = WaterControl.objective(solve_forward_2D(prob, a_plus), prob.target)
        obj_minus = WaterControl.objective(solve_forward_2D(prob, a_minus), prob.target)
        
        gradients[idx] = (obj_plus - obj_minus) / (2s)
    end
    
    return gradients
end


@testset "Gradient Tests" begin
    @testset "Direct vs Adjoint 1D" begin
        for _ in 1:10
            nx = 100
            target = 1 .+ randn(nx)
            prob = ShallowWaterProblem1D(target)
            max_time = prob.tspan[2]
            acc_points = 50
            acc = DiscreteAcceleration1D(zeros(acc_points), LinRange(0, max_time, acc_points + 1))

            direct_grad = gradient(prob, acc)
            adjoint_grad = WaterControl.adjoint_gradient(prob, acc)

            @test isapprox(direct_grad, adjoint_grad, rtol=1e-1)
        end
    end

    @testset "Direct vs Adjoint 2D" begin
        for _ in 1:10
            nx, ny = 100, 100
            target = 1 .+ randn(nx, ny)
            prob = ShallowWaterProblem2D(target)
            max_time = prob.tspan[2]
            acc_points = 50
            acc = DiscreteAcceleration2D(
                zeros(acc_points),
                zeros(acc_points),
                LinRange(0, max_time, acc_points + 1)
            )

            direct_grad = gradient(prob, acc)
            adjoint_grad = WaterControl.adjoint_gradient(prob, acc)

            @test isapprox(direct_grad, adjoint_grad, rtol=1e-1)
        end
    end
end


# testing the mesh for completeness
# for he in mesh.edges
#     if he.twin === nothing
#         println("twin missing", he)
#     end
#     if he.next === nothing
#         println("next missing", he)
#     end
#     if he.prev === nothing
#         println("prev missing", he)
#     end
# end

# for i in 1:height+1, j in 1:width+1
#     if !isassigned(mesh.points, i, j)
#         println("no he assigned", i, j)
#     end
# end