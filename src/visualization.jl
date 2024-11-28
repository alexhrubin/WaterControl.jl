using Plots
using Printf

# 1D animation
function create_animation(sol, prob, fps=20)
    final_time = sol.t[end]
    n_frames = Int(final_time * fps)
    frame_steps = LinRange(0, final_time, n_frames)

    nx = prob.nx
    max = maximum(maximum(u[1:nx]) for u in sol.u)

    p = plot(ylim=(0, max*1.05))

    anim = @animate for t in frame_steps
        h = sol(t)[1:nx]

        empty!(p[1])

        plot!(p[1],
            prob.x,
            h,
            color=1,
            title="Water Surface at t = $(@sprintf("%.2f", t)) s",
            xlabel="Position (m)",
            ylabel="Height (m)",
            titlefont=font("Monaco", 12),
            tickfont=font("Monaco", 10),
            guidefont=font("Monaco", 11),  # for axis labels
            legend=false,
        )
    end

    return gif(anim)
end
