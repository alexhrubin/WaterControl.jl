using LinearAlgebra

"""
    solve_poisson_sor!(Φ, L, h=1.0; omega=1.5, epsilon=1e-6, max_iter=10000)

Solve the Poisson equation ∇²Φ = L using Successive Over-Relaxation.
Modifies Φ in-place for better performance.

Parameters:
- Φ: solution array (will be modified in-place)
- L: source term matrix
- h: grid spacing
- omega: relaxation parameter (1 < omega < 2)
- epsilon: convergence criterion
- max_iter: maximum number of iterations

Returns:
- iterations: number of iterations performed
- error: final error
"""
function solve_poisson_sor!(Φ::Matrix{Float64}, L::Matrix{Float64}, h::Float64=1.0;
                          omega::Float64=1.5, epsilon::Float64=1e-6, max_iter::Int=10000)
    n, m = size(L)
    h2 = h * h
    error = Inf
    iterations = 0
    
    # Pre-allocate temporary variable for thread safety
    new_value = 0.0
    
    while error > epsilon && iterations < max_iter
        error = 0.0
        
        # Red-Black ordering for better parallelization
        for color in 0:1  # 0=red, 1=black
            # Use @inbounds for better performance since we check bounds manually
            @inbounds for i in 2:(n-1)
                start = 2 + (i + color) % 2
                for j in start:2:(m-1)
                    old_value = Φ[i,j]
                    
                    # Calculate new value
                    new_value = 0.25 * (
                        Φ[i+1,j] + Φ[i-1,j] + 
                        Φ[i,j+1] + Φ[i,j-1] - 
                        h2 * L[i,j]
                    )
                    
                    # Apply SOR
                    Φ[i,j] = old_value + omega * (new_value - old_value)
                    
                    # Track maximum error
                    error = max(error, abs(new_value - old_value))
                end
            end
        end
        
        iterations += 1
        
        # Optional: Print progress every 100 iterations
        if iterations % 100 == 0
            @printf("Iteration %d, Error: %.2e\n", iterations, error)
        end
    end
    
    return iterations, error
end


function solve_poisson_sor(L::Matrix{Float64}, h::Float64=1.0; kwargs...)
    Φ = zeros(size(L))
    iterations, error = solve_poisson_sor!(Φ, L, h; kwargs...)
    return Φ
end
