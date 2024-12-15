using WriteVTK

mutable struct Point
    x::Float64
    y::Float64
    i::Union{Int, Nothing}  # Image row index
    j::Union{Int, Nothing}  # Image column index

    Point(x::Float64, y::Float64) = new(x, y, nothing, nothing)
    Point(x::Float64, y::Float64, i::Int, j::Int) = new(x, y, i, j)
end


mutable struct HalfEdge
    origin::Point
    next::Union{Int, Nothing}
    prev::Union{Int, Nothing}
    twin::Union{Int, Nothing}
    face::Union{Int, Nothing}
    
    HalfEdge(origin::Point) = new(origin, nothing, nothing, nothing)
end


struct Face 
    edge::Int  # Reference to any half-edge of this face
    i::Int  # Image row index
    j::Int  # Image column index
    value::Float64  # value of the pixel[i, j] in the original image
end


struct Mesh
    points::Matrix{Point}
    edges::Vector{HalfEdge}
    point_edge::Matrix{HalfEdge}
    faces::Vector{Face}
end


function make_mesh(img::Matrix{Float64}, physical_height::Float64, physical_width::Float64)
    height, width = size(img)
    pixel_height = physical_height / height
    pixel_width = physical_width / width
    
    # create array of disconnected Points
    points = Matrix{Point}(undef, height + 1, width + 1)
    for j in 0:width, i in 0:height
        points[i+1, j+1] = Point(j * pixel_width, i * pixel_height, i+1, j+1)
    end

    edge_map = Dict{Tuple{Point,Point}, Int}()
    point_edge = Matrix{HalfEdge}(undef, height+1, width+1)  #Dict{Point, Int}()
    edges = HalfEdge[]
    faces = Face[]

    function add_edge(point)
        he = HalfEdge(point)
        push!(edges, he)
        en = length(edges)
        return en
    end

    # clockwise within each triangle from the top left corner
    for j in 1:width, i in 1:height
        en1 = add_edge(points[i, j])
        en2 = add_edge(points[i, j+1])
        en3 = add_edge(points[i+1, j])

        edges[en1].next = en2
        edges[en2].next = en3
        edges[en3].next = en1
        
        edges[en1].prev = en3
        edges[en2].prev = en1
        edges[en3].prev = en2
        
        edge_map[(edges[en1].origin, edges[en2].origin)] = en1
        edge_map[(edges[en2].origin, edges[en3].origin)] = en2
        edge_map[(edges[en3].origin, edges[en1].origin)] = en3
        
        face = Face(en1, i, j, img[i, j])
        push!(faces, face)
        edges[en1].face = length(faces)
        edges[en2].face = length(faces)
        edges[en3].face = length(faces)

        point_edge[i, j] = edges[en1]
    end

    # clockwise within each triangle from the bottom right corner
    for j in 2:width+1, i in 2:height+1
        en1 = add_edge(points[i, j])
        en2 = add_edge(points[i, j-1])
        en3 = add_edge(points[i-1, j])

        edges[en1].next = en2
        edges[en2].next = en3
        edges[en3].next = en1

        edges[en1].prev = en3
        edges[en2].prev = en1
        edges[en3].prev = en2

        edge_map[(edges[en1].origin, edges[en2].origin)] = en1
        edge_map[(edges[en2].origin, edges[en3].origin)] = en2
        edge_map[(edges[en3].origin, edges[en1].origin)] = en3

        face = Face(en1, i, j, img[i-1, j-1])
        push!(faces, face)
        edges[en1].face = length(faces)
        edges[en2].face = length(faces)
        edges[en3].face = length(faces)

        if !isassigned(point_edge, i, j)
            point_edge[i, j] = edges[en1]
        end
    end
    
    # add edges counterclockwise around the outside
    edge_points = [
        points[:, 1]...,
        points[height+1, 2:end]...,
        points[end-1:-1:1, width+1]...,
        points[1, end-1:-1:2]...
    ]
    en1 = add_edge(edge_points[1])
    first = en1
    
    for ep in edge_points[2:end]
        en2 = add_edge(ep)
        edges[en1].next = en2
        edges[en2].prev = en1
        edge_map[(edges[en1].origin, edges[en2].origin)] = en1
        en1 = en2
    end
    edges[first].prev = length(edges)
    edges[length(edges)].next = first
    edge_map[(edges[length(edges)].origin, edges[first].origin)] = length(edges)

    # pair up edge twins
    for (p1, p2) in keys(edge_map)
        edge_id = edge_map[(p1, p2)]
        edges[edge_id].twin = edge_map[(p2, p1)]
    end

    # finally, top right and bottom left points need associated HalfEdges
    # we'll get them as twins of edges which terminate at the corners
    idx = point_edge[1, width].twin
    point_edge[1, width+1] = edges[idx]
    
    idx = point_edge[height+1, 2].twin
    point_edge[height+1, 1] = edges[idx]

    return Mesh(points, edges, point_edge, faces)
end


function traverse_triangle(point, mesh, direction=:clockwise)
    if direction == :clockwise
        he = mesh.point_edge[point.i, point.j]
    else
        he = mesh.edges[mesh.point_edge[point.i, point.j].twin]
    end

    points = [he.origin]
    next_en = nothing
    for _ in 1:3
        next_en = he.next
        push!(points, mesh.edges[next_en].origin)
        he = mesh.edges[next_en]
    end
    return points
end

function iterate_around_vertex(point, mesh)
    he1 = mesh.point_edge[point.i, point.j]
    hes = HalfEdge[he1]
    he2 = nothing
    for _ in 1:6
        he2 = mesh.edges[mesh.edges[he1.prev].twin]
        if he2 == first(hes)
            break
        end
        push!(hes, he2)
        he1 = he2
    end
    return hes
end


function midpoint(p1::Point, p2::Point)
    return Point((p1.x + p2.x)/2, (p1.y + p2.y)/2)
end


function centroid(point_1::Point, point_2::Point, point_3::Point)
    x = (point_1.x + point_2.x + point_3.x) / 3
    y = (point_1.y + point_2.y + point_3.y) / 3
    return Point(x, y)
end


function _other_two_hes(he1, mesh)
    he2 = mesh.edges[he1.next]
    he3 = mesh.edges[he2.next]
    return he2, he3
end


function dual_cell(he1, mesh)
    points = Point[]
    he2, he3 = _other_two_hes(he1, mesh)

    mp = midpoint(he1.origin, he2.origin)
    push!(points, mp)
    
    cent = centroid(he1.origin, he2.origin, he3.origin)
    push!(points, cent)

    mp = midpoint(he1.origin, he3.origin)
    push!(points, mp)

    push!(points, he1.origin)
    push!(points, points[1])  # to close the polygon

    return points
end


function quad_area(points::Vector{Point})
    x = map(p -> p.x, points)
    y = map(p -> p.y, points)

    # Uses shoelace formula
    return 0.5 * abs(
        x[1]*y[2] + x[2]*y[3] + x[3]*y[4] + x[4]*y[1] -
        y[1]*x[2] - y[2]*x[3] - y[3]*x[4] - y[4]*x[1]
    )
 end


function dual_mesh_value(point, mesh)
    hes = WaterControl.iterate_around_vertex(point, mesh)
    hes = filter(he -> he.face !== nothing, hes)

    areas = []
    face_values = []
    for he in hes
        pts = WaterControl.dual_cell(he, mesh)
        area = quad_area(pts)
        push!(areas, area)

        face = mesh.faces[he.face]
        push!(face_values, face.value)
    end

    return sum(face_values .* areas) / sum(areas)
end


function dual_mesh_area(point::Point, mesh::Mesh)
    hes = WaterControl.iterate_around_vertex(point, mesh)
    hes = filter(he -> he.face !== nothing, hes)

    areas = []
    for he in hes
        pts = WaterControl.dual_cell(he, mesh)
        area = quad_area(pts)
        push!(areas, area)
    end

    return sum(areas)
end


function dual_mesh_areas(mesh::Mesh)
    height, width = size(mesh.points)
    dual_areas = Matrix{Float64}(undef, height, width)
    for point in mesh.points
        dual_areas[point.i, point.j] = dual_mesh_area(point, mesh)
    end

    # Edges (not corners)
    dual_areas[1, 2:end-1] .*= 2  # top
    dual_areas[end, 2:end-1] .*= 2  # bottom
    dual_areas[2:end-1, 1] .*= 2  # left
    dual_areas[2:end-1, end] .*= 2  # right
    
    # Corners
    dual_areas[1, 1] *= 6  # top-left
    dual_areas[1, end] *= 3  # top-right
    dual_areas[end, 1] *= 3  # bottom-left
    dual_areas[end, end] *= 6  # bottom-right

    return dual_areas
end


function dual_mesh_values(mesh)
    height, width = size(mesh.points)
    dual_vals = Matrix{Float64}(undef, height, width)
    for point in mesh.points
        dual_vals[point.i, point.j] = WaterControl.dual_mesh_value(point, mesh)
    end
    return dual_vals
end


function save_mesh(mesh::Mesh, filename::String)
    points_flat = vec(mesh.points)
    point_to_idx = Dict(p => i for (i,p) in enumerate(points_flat))
    
    points_array = zeros(3, length(points_flat))
    for (i, p) in enumerate(points_flat)
        points_array[1,i] = p.x
        points_array[2,i] = p.y
        points_array[3,i] = 0.0
    end

    cells = MeshCell[]
    face_values = Float64[]
    for face in mesh.faces
        edge = mesh.edges[face.edge]
        indices = Int[
            point_to_idx[edge.origin],
            point_to_idx[mesh.edges[edge.next].origin],
            point_to_idx[mesh.edges[mesh.edges[edge.next].next].origin]
        ]
        push!(cells, MeshCell(VTKCellTypes.VTK_TRIANGLE, indices))
        push!(face_values, face.value)
    end

    vtk_grid(filename, points_array, cells) do vtk
        vtk_cell_data(vtk, face_values, "value")
        vtk_save(vtk)
    end
end
