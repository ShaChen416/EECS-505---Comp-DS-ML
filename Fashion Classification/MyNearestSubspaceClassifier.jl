module MyNearestSubspaceClassifier

#TODO: put all the "learning" functions necessary for running the nearest subspace based classification algorithm 
println("Initializing MyNearestSubspaceClassifier module ... ")
export  linear, dlinear, dtanh, learn_nearest_ss, classify_nearest_ss # Export the functions needed ##
end

function linear(z)
    return z
end

function dlinear(z)
    return 1.0  
   
end

function dtanh_sol(z)
return 1-tanh(z)^2
end

#TODO: put all the "learning" functions necessary for running the nearest subspace classifier 

using LinearAlgebra: svd
function learn_nearest_ss(train::Array, ktrain::Integer)
    n, N, d = size(train)

    # Compute and save basis vectors
    U = zeros(n, ktrain, d)
    for j in 1:d
          Uj = svd(train[:,:,j]).U  # TODO: fill in ??
          U[:, :, j] = Uj[:, 1:ktrain, :] # TODO: fill in ??
    end
    return U
end


function classify_nearest_ss(test::Matrix, U::Array, k::Integer=size(U, 2))
    n, t = size(test)
    d = size(U, 3)
    
    # Construct projection matrices
    P = zeros(n, n, d)
    for j in 1:d
        Uk = U[:, 1:k, j]
        P[:, :, j] =  Uk*Uk'  # TODO: Fill in ??. Hint: How is U organized?
    end

    # Calculate projection errors
    err = zeros(d, t)
    for j in 1:d
        err[j, :] = sum((test - P[:, :, j] * test).^2; dims=1)  
    end
    
    # Classify each vector by which subspace is nearest
    labels = mapslices(x -> findmin(x)[2], err; dims=1) |> vec   
    return labels
end


