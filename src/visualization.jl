using Plots
using Printf

function create_animation_1D(sol, prob, fps=20)
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


function create_surface_animation_2D(sol, prob, fps=20)
    nx = prob.nx
    ny = prob.ny
    final_time = sol.t[end]
    n_frames = Int(final_time * fps)
    frame_steps = LinRange(0, final_time, n_frames)
    
    min_h = minimum(minimum(u[1:nx*ny]) for u in sol.u)
    max_h = maximum(maximum(u[1:nx*ny]) for u in sol.u)

    p = plot(size=(500, 500))

    anim = @animate for t in frame_steps
        surf = reshape(sol(t)[1:nx*ny], ny, nx)

        empty!(p[1])
        
        surface!(
            p[1],
            prob.x,
            prob.y,
            surf, 
            camera=(45, 45),
            color=:viridis,
            colorbar=false,
            xlabel="X Position (m)",
            ylabel="Y Position (m)",
            zlabel="Height (m)",
            title="Water Surface at t = $(@sprintf("%.2f", t)) s",
            zlim=(min_h, max_h),
            clim=(min_h, max_h),
            titlefont=font("Monaco", 15),
            tickfont=font("Monaco", 10),
            guidefont=font("Monaco", 11),
        )
    end

    return gif(anim)
end


function create_contour_animation_2D(sol, prob, fps=20)
    nx = prob.nx
    ny = prob.ny
    final_time = sol.t[end]
    n_frames = Int(final_time * fps)
    frame_steps = LinRange(0, final_time, n_frames)

    p = plot(size=(500, 500))

    min_h = minimum(minimum(u[1:nx*ny]) for u in sol.u)
    max_h = maximum(maximum(u[1:nx*ny]) for u in sol.u)

    anim = @animate for t in frame_steps
        surf = reshape(sol(t)[1:nx*ny], ny, nx)

        empty!(p[1])

        contour!(
            p[1],
            prob.x,
            prob.y,
            surf,
            fill=true,
            aspect_ratio=:equal,
            clim=(min_h, max_h),
        )
    end

    return gif(anim)
end
