using ShallowWater
using Test


# compute gradient from multiple forward solves.
function gradient(prob, acc)
    objectives_plus = []
    s = 0.01
    for idx in 1:length(acc.values)
        a = DiscreteAcceleration(collect(acc.values), collect(LinRange(0, 10, length(acc.times))))
        a[idx] += s
        sol = solve_forward(prob, a)
        obj = ShallowWater.objective(sol, prob.target)
        push!(objectives_plus, obj)
    end
    
    objectives_minus = []
    for idx in 1:length(acc.values)
        a = DiscreteAcceleration(collect(acc.values), collect(LinRange(0, 10, length(acc.times))))
        a[idx] += -s
        sol = solve_forward(prob, a)
        obj = ShallowWater.objective(sol, prob.target)
        push!(objectives_minus, obj)
    end

    return (objectives_plus - objectives_minus) ./ 2s
end


@testset "Gradient Tests" begin
    @testset "Direct vs Adjoint" begin
        for _ in 1:10
            nx = 100
            # target = 1 .+ [0.2*sin(2i * pi / nx) for i in 1:nx]
            target = 1 .+ randn(nx)
            prob = ShallowWaterProblem(target)
            max_time = prob.tspan[2]
            acc_points = 50
            acc = DiscreteAcceleration(zeros(acc_points), LinRange(0, max_time, acc_points + 1))

            direct_grad = gradient(prob, acc)
            adjoint_grad = ShallowWater.adjoint_gradient(prob, acc)

            @test isapprox(direct_grad, adjoint_grad, rtol=1e-1)
        end
    end
end
