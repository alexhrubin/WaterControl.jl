using WaterControl
using LinearAlgebra

function get_face_nodes(face::Face, mesh::Mesh)
    nodes = Point[]
    start_edge = face.edge
    current_edge = start_edge
    
    while true
        push!(nodes, mesh.edges[current_edge].origin)
        current_edge = mesh.edges[current_edge].next
        if current_edge == start_edge || isnothing(current_edge)
            break
        end
    end
    
    return nodes
end

function is_boundary_edge(edge::HalfEdge, mesh::Mesh)
    return isnothing(edge.twin)
end

function get_boundary_edges(mesh::Mesh)
    return findall(e -> is_boundary_edge(e, mesh), mesh.edges)
end

function compute_edge_normal(edge::HalfEdge, mesh::Mesh)
    p1 = edge.origin
    p2 = mesh.edges[edge.next].origin
    
    dx = p2.x - p1.x
    dy = p2.y - p1.y
    
    length = sqrt(dx^2 + dy^2)
    return [-dy/length, dx/length]
end

function assemble_system(mesh::Mesh, L::Matrix{Float64}, g::Function)
    # Get total number of points
    n_points = length(mesh.points)
    
    # Initialize dense matrices
    K = zeros(n_points, n_points)
    b = zeros(n_points)
    
    # Assemble contributions from each face
    for face in mesh.faces
        nodes = get_face_nodes(face, mesh)
        n_nodes = length(nodes)
        
        # Skip if we don't have a triangle
        if n_nodes != 3
            continue
        end
        
        # Get node indices in the global system
        node_indices = [LinearIndices(mesh.points)[findfirst(p -> p == node, mesh.points)] for node in nodes]
        
        # Compute element matrices
        xe = [n.x for n in nodes]
        ye = [n.y for n in nodes]
        
        # Compute Jacobian
        J = [xe[2]-xe[1] xe[3]-xe[1]
             ye[2]-ye[1] ye[3]-ye[1]]
        detJ = abs(det(J))
        
        # Shape function derivatives in reference coordinates
        dN = Float64[-1 -1;
                      1  0;
                      0  1]
        
        # Transform to physical coordinates
        B = dN * inv(J)  # B is 3×2

        # Element stiffness matrix
        Ke = zeros(3,3)
        for i in 1:3
            for j in 1:3
                Ke[i,j] = detJ * dot(B[i,:], B[j,:]) / 2
            end
        end
        
        # Element load vector (source term)
        fe = detJ * L[face.i, face.j] * ones(3) / 3
        
        # Assembly into global system
        for i in 1:3
            ni = node_indices[i]
            b[ni] += fe[i]
            for j in 1:3
                nj = node_indices[j]
                K[ni, nj] += Ke[i,j]
            end
        end
    end
    
    # Assemble Neumann boundary conditions
    for edge_idx in get_boundary_edges(mesh)
        edge = mesh.edges[edge_idx]
        
        p1 = edge.origin
        p2 = mesh.edges[edge.next].origin
        
        length = sqrt((p2.x - p1.x)^2 + (p2.y - p1.y)^2)
        
        xmid = (p1.x + p2.x)/2
        ymid = (p2.y + p2.y)/2
        
        normal = compute_edge_normal(edge, mesh)
        gval = g(xmid, ymid)
        
        n1 = findfirst(p -> p == p1, mesh.points)
        n2 = findfirst(p -> p == p2, mesh.points)
        
        if !isnothing(n1) && !isnothing(n2)
            b[n1] += length * gval * normal / 2
            b[n2] += length * gval * normal / 2
        end
    end
    
    # Add small regularization for pure Neumann problem
    K += I * 1e-6
    
    return K, b
end

function solve_poisson_fem(mesh::Mesh, L::Matrix{Float64}, g::Function)
    # Assemble system
    K, b = assemble_system(mesh, L, g)
    
    # Solve using direct dense solver
    Φ = K \ b
    
    return Φ
end


##########

using LinearAlgebra
using SparseArrays

function solve_poisson_fem_sparse(mesh::Mesh, L::Matrix{Float64}, g::Function)
    n_faces = length(mesh.faces)
    n_points = length(mesh.points)
    
    # Vectors for sparse matrix construction
    rows = Int[]
    cols = Int[]
    vals = Float64[]
    b = zeros(n_points)
    
    dN = Float64[-1 -1;
                  1  0;
                  0  1]
    
    for (face_idx, face) in enumerate(mesh.faces)
        nodes = get_face_nodes(face, mesh)
        n_nodes = length(nodes)
        
        n_nodes == 3 || continue
        
        node_indices = [LinearIndices(mesh.points)[findfirst(p -> p == node, mesh.points)] for node in nodes]
        
        # Get coordinates
        x1, y1 = nodes[1].x, nodes[1].y
        x2, y2 = nodes[2].x, nodes[2].y
        x3, y3 = nodes[3].x, nodes[3].y
        
        # Compute Jacobian matrix
        Jmat = [x2-x1 x3-x1
                y2-y1 y3-y1]
        detJ = det(Jmat)
        
        if abs(detJ) < 1e-10
            @warn "Nearly degenerate triangle found at face $face_idx"
            continue
        end
        
        # Transform to physical coordinates
        B = dN * inv(Jmat)
        
        # Compute element stiffness matrix
        Ke = zeros(3,3)
        for i in 1:3
            for j in 1:3
                Ke[i,j] = abs(detJ) * dot(B[i,:], B[j,:]) / 2
            end
        end
        
        # Element load vector
        f_val = L[face.i, face.j] * abs(detJ) / 6
        
        # Assembly
        for i in 1:3
            ni = node_indices[i]
            b[ni] += f_val
            
            for j in 1:3
                nj = node_indices[j]
                push!(rows, ni)
                push!(cols, nj)
                push!(vals, Ke[i,j])
            end
        end
    end
    
    K = sparse(rows, cols, vals, n_points, n_points)
    
    # Add Neumann boundary conditions
    for edge_idx in findall(e -> is_boundary_edge(e, mesh), mesh.edges)
        edge = mesh.edges[edge_idx]
        p1 = edge.origin
        p2 = mesh.edges[edge.next].origin
        
        length = sqrt((p2.x - p1.x)^2 + (p2.y - p1.y)^2)
        xmid = (p1.x + p2.x)/2
        ymid = (p2.y + p2.y)/2
        
        n1 = LinearIndices(mesh.points)[findfirst(p -> p == p1, mesh.points)]
        n2 = LinearIndices(mesh.points)[findfirst(p -> p == p2, mesh.points)]
        
        normal = compute_edge_normal(edge, mesh)
        gval = g(xmid, ymid)
        
        b[n1] += length * gval * normal / 2
        b[n2] += length * gval * normal / 2
    end
    
    # Add regularization for pure Neumann problem
    K = K + spdiagm(0 => fill(1e-6, n_points))
    
    # Solve system
    Φ = K \ b
    
    return Φ
end
