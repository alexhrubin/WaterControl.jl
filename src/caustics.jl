struct SquareMesh
    xs::Vector{Float64}
    ys::Vector{Float64}
    values::Matrix{Float64}

    function SquareMesh(xs, ys)
        values = zeros(Float64, length(xs)-1, length(ys)-1)
        return new(xs, ys, values)
    end
end


function make_square_mesh(Lx, Ly, nx, ny)
    xs = range(0, Lx, length=nx)
    ys = range(0, Ly, length=ny)
    return SquareMesh(xs, ys)
end


function in_mesh_at_all(point, square_mesh)
    return point.x >= square_mesh.xs[1] && point.x <= square_mesh.xs[end] &&
           point.y >= square_mesh.ys[1] && point.y <= square_mesh.ys[end]
end


function in_cell(point, square_mesh::SquareMesh)
    i = findfirst(y -> y >= point.y, square_mesh.ys)
    j = findfirst(x -> x >= point.x, square_mesh.xs)
    return i-1, j-1
end


function collect_values(points, square_mesh::SquareMesh)
    points_in_mesh = filter(p -> in_mesh_at_all(p, square_mesh), points)
    for p in points_in_mesh
        i, j = in_cell(p, square_mesh)
        square_mesh.values[i, j] += 1
    end
    return square_mesh
end
