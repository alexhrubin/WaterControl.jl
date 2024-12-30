function compute_normal_angles(p1, p2, p3)
    # Compute vectors along two edges of the triangle
    v1 = [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]
    v2 = [p3.x - p1.x, p3.y - p1.y, p3.z - p1.z]
    
    # Cross product gives normal vector
    normal = [
        v1[2]*v2[3] - v1[3]*v2[2],
        v1[3]*v2[1] - v1[1]*v2[3], 
        v1[1]*v2[2] - v1[2]*v2[1]
    ]
    
    # Normalize to unit vector
    magnitude = sqrt(sum(normal.^2))
    normal = normal ./ magnitude
    
    # Calculate angles relative to z axis
    # angle_x is angle between normal's projection on xz plane and z axis
    # angle_y is angle between normal's projection on yz plane and z axis
    angle_x = atan(normal[1], normal[3])  # atan2 but we know z is positive
    angle_y = atan(normal[2], normal[3])
    
    # Apply Snell's law: n1 * sin(θ1) = n2 * sin(θ2)
    # Here n1 = 1 (air) and n2 = 1.5 (material)
    refracted_x = asin(sin(angle_x) / 1.33)
    refracted_y = asin(sin(angle_y) / 1.33)
    return refracted_x, refracted_y
end


function create_grid_points(mesh::Mesh, n)
    # Find bounds of mesh
    x_min = minimum(p.x for p in mesh.points)
    x_max = maximum(p.x for p in mesh.points)
    y_min = minimum(p.y for p in mesh.points)
    y_max = maximum(p.y for p in mesh.points)
    
    # Create grid
    x_range = range(x_min, x_max, length=n)
    y_range = range(y_min, y_max, length=n)
    return [(x, y) for x in x_range, y in y_range]
end


function point_in_triangle(px, py, p1::Point, p2::Point, p3::Point, ε=1e-10)
    area = 0.5 * (-p2.y * p3.x + p1.y * (-p2.x + p3.x) + p1.x * (p2.y - p3.y) + p2.x * p3.y)
    s = 1/(2*area) * (p1.y * p3.x - p1.x * p3.y + (p3.y - p1.y) * px + (p1.x - p3.x) * py)
    t = 1/(2*area) * (p1.x * p2.y - p1.y * p2.x + (p1.y - p2.y) * px + (p2.x - p1.x) * py)
    
    return s >= -ε && t >= -ε && (1-s-t) >= -ε
end


function find_containing_triangle(point, mesh::Mesh)
    for (i, face) in enumerate(mesh.faces)
        points = traverse_triangle(face, mesh)[1:end-1]
        if point_in_triangle(point[1], point[2], points...)
            return i
        end
    end
    return nothing
end


function get_plane_intersection(px::Float64, py::Float64, p1::Point, p2::Point, p3::Point)
    v1 = [p2.x - p1.x, p2.y - p1.y, p2.z - p1.z]
    v2 = [p3.x - p1.x, p3.y - p1.y, p3.z - p1.z]
    
    normal = [
        v1[2]*v2[3] - v1[3]*v2[2],
        v1[3]*v2[1] - v1[1]*v2[3], 
        v1[1]*v2[2] - v1[2]*v2[1]
    ]
    
    d = -(normal[1]*p1.x + normal[2]*p1.y + normal[3]*p1.z)
    z = -(normal[1]*px + normal[2]*py + d) / normal[3]
    
    return z
end


# function project_grid(mesh::Mesh, n::Int)
#     grid_points = create_grid_points(mesh, n)
#     projections = Matrix{Union{Nothing, Point}}(nothing, size(grid_points)...)
#     zs = fill(NaN, size(grid_points)...)  # Initialize with NaN to catch unset values

#     for idx in CartesianIndices(grid_points)
#         x, y = grid_points[idx]
#         face_idx = find_containing_triangle((x,y), mesh)
        
#         if !isnothing(face_idx)
#             face = mesh.faces[face_idx]
#             points = traverse_triangle(face, mesh)[1:end-1]
#             z = get_plane_intersection(x, y, points...)
#             zs[idx] = z

#             angle_x, angle_y = compute_normal_angles(points[1], points[2], points[3])
#             projections[idx] = Point(
#                 x - z * sin(angle_x),
#                 y - z * sin(angle_y),
#             )
#         end
#     end
#     return projections, zs
# end


function project_grid(mesh::Mesh, n::Int, Lx::Real, Ly::Real)
    # Initialize outputs
    projections = Matrix{Union{Nothing, Point}}(nothing, n, n)
    zs = fill(NaN, n, n)
    
    # Create spatial index of triangles
    # Store triangles in grid cells they overlap with
    grid_cells = Dict{Tuple{Int,Int}, Vector{Face}}()
    ncells = max(1, floor(Int, sqrt(length(mesh.faces)/4)))  # Adjust cell count based on triangle count
    
    # Populate spatial index
    for face in mesh.faces
        points = traverse_triangle(face, mesh)[1:end-1]
        x_min = minimum(p.x for p in points)
        x_max = maximum(p.x for p in points)
        y_min = minimum(p.y for p in points)
        y_max = maximum(p.y for p in points)
        
        # Find overlapping cells
        i_min = max(1, floor(Int, y_min * ncells/Ly))
        i_max = min(ncells, ceil(Int, y_max * ncells/Ly))
        j_min = max(1, floor(Int, x_min * ncells/Lx))
        j_max = min(ncells, ceil(Int, x_max * ncells/Lx))
        
        for i in i_min:i_max, j in j_min:j_max
            key = (i,j)
            if !haskey(grid_cells, key)
                grid_cells[key] = Face[]
            end
            push!(grid_cells[key], face)
        end
    end
    
    # Project rays using spatial index
    for i in 1:n, j in 1:n
        x = (j - 1) * (Lx/(n-1))
        y = (i - 1) * (Ly/(n-1))
        
        # Find which cell this point is in
        cell_i = max(1, min(ncells, floor(Int, y * ncells/Ly)))
        cell_j = max(1, min(ncells, floor(Int, x * ncells/Lx)))
        
        # Only check triangles in this cell
        if haskey(grid_cells, (cell_i, cell_j))
            for face in grid_cells[(cell_i, cell_j)]
                points = traverse_triangle(face, mesh)[1:end-1]
                if point_in_triangle(x, y, points...)
                    z = get_plane_intersection(x, y, points...)
                    angle_x, angle_y = compute_normal_angles(points...)
                    
                    zs[i,j] = z
                    projections[i,j] = Point(
                        x - z * sin(angle_x),
                        y - z * sin(angle_y),
                    )
                    break
                end
            end
        end
    end
    
    return projections, zs
end


function compute_caustics(projected_grid::Matrix, resolution::Int, Lx::Real, Ly::Real)
    caustic_buffer = zeros(resolution, resolution)
    radius_pixels = 30  # Maximum radius to consider in pixels
    
    for p in projected_grid
        if !isnothing(p)
            # Scale physical coordinates to unit square for pixel calculation
            x_scaled = p.x / Lx
            y_scaled = p.y / Ly
            
            # Get pixel coordinates of photon hit
            ci = clamp(floor(Int, y_scaled * resolution) + 1, 1, resolution)
            cj = clamp(floor(Int, x_scaled * resolution) + 1, 1, resolution)
            
            # Only compute for pixels within radius
            imin = max(1, ci - radius_pixels)
            imax = min(resolution, ci + radius_pixels)
            jmin = max(1, cj - radius_pixels)
            jmax = min(resolution, cj + radius_pixels)
            
            # Create coordinate arrays for this region (in physical coordinates)
            y_coords = reshape(((imin:imax) .- 0.5) * (Ly/resolution), :, 1)
            x_coords = reshape(((jmin:jmax) .- 0.5) * (Lx/resolution), 1, :)
            
            # Compute distances in physical space
            R2 = (y_coords .- p.y).^2 .+ (x_coords .- p.x).^2
            
            # Scale σ to physical dimensions (assuming σ should be proportional to domain size)
            σ = 0.015 * sqrt(Lx * Ly)  # Scale with geometric mean of dimensions
            
            # Update buffer view
            @views caustic_buffer[imin:imax, jmin:jmax] .+= 0.3 * exp.(-R2/(2σ^2))
        end
    end
    
    return caustic_buffer
end


function update_mesh_heights!(mesh::Mesh, surf::Matrix{Float64})
    # Update z coordinate of each Point in the mesh
    # Note: mesh.points is (height+1, width+1) while surf is (height, width)
    height, width = size(surf)
    
    # For interior points (where we have surf values)
    for i in 1:height, j in 1:width
        mesh.points[i,j].z = surf[i,j]
    end
    
    # For edge points, interpolate from neighbors
    for j in 1:width
        mesh.points[height+1,j].z = surf[height,j]
    end
    for i in 1:height
        mesh.points[i,width+1].z = surf[i,width]
    end
    mesh.points[height+1,width+1].z = surf[height,width]
end
