function tv_admm(y, A, C, λ, ρ, nIters)
    # Precompute constants
    P = A'A + ρ*(C'C)
    Aty = A'y

    # ADMM updates
    x = zeros(size(A,2))
    z = zeros(size(C,1))
    u = zeros(size(C,1))
    for _ = 1:nIters
        # x update
        x = P \ (Aty + ρ*(C'*(z - u)))
        Cx = C*x

        # z update
        z = soft.(Cx + u, (λ/2) / ρ)

        # u update
        u = u + Cx - z
    end

    return x
end
soft(x,mu) = sign(x)*max(abs(x)-mu,0)