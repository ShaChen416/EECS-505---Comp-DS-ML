module MyLearn2ClassifyMany 

#TODO: put all the "learning" functions necessary for running the algorithms Learn2ClassifyMany

println("Initializing MyLearn2ClassifyMany module ... ")
## Activation functions
export linear, dlinear, dtanh, grad_loss_1layer, g, grad_loss_1layer_1output, learn2classify_asgd_1layer, load_item_data #, g, grad_loss_1layer, grad_loss_1layer_1output ## 
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

function g(X::Array, W::Matrix, b::Vector, f_a::Function)
    return sum(f_a.(W * X .+ b); dims=1) # type Array{Float64,2}
end


#TODO: put all the "learning" functions necessary for running the algorithms Learn2ClassifyMany
function grad_loss_1layer(
        f_a::Function,
        df_a::Function, 
        x::Matrix, 
        y::Matrix, 
        W::Matrix,
        b::Vector
    )

    n, d = size(W) ##TODO: ?? & quiz
    N = size(y,2) ## assume y is matrix of size n x N
    
    dW = zeros(n, d) 
    db = zeros(n)
    loss = 0.0

    for k in 1:N
        for p in 1:n
            error = y[p, k] - f_a(W[p, :]' * x[:, k] + b[p])
            common_term = error * df_a(W[p, :]' * x[:, k] + b[p])
            for q in 1:d
                dW[p, q] = dW[p, q] - 2 / N * common_term * x[q, k]
            end
            db[p] = db[p] - 2 / N * common_term * 1
            loss = loss + 1 / N * error^2
        end
    end
    return dW, db, loss
end

function grad_loss_1layer_1output(
        f_a::Function,
        df_a::Function,
        x::Matrix,
        y::Matrix,
        W::Matrix,
        b::Vector
        )          
    
    n, d = size(W) 
    N = size(x, 2)
    
    dW = zeros(n, d)
    db = zeros(n)
    loss = 0.0
    
    for k in 1 : N
        error = (y[k] - sum(f_a.(W * x[:, k] + b)))
        for p in 1 : n
            for q in 1 : d
                # TODO: Fill in the ??
                dW[p, q] = dW[p, q] - 2 / N * error * df_a(W[p, :]' * x[: ,k] + b[p]) * x[q, k]
            end
            #TODO: Fill in the ??
            db[p] = db[p] - 2 / N * error * df_a(W[p, :]' * x[:, k] + b[p])
        end
        
        ## TODO: Fill in the ??
        loss =  loss + 1/N * error^2
    end
    return dW, db, loss
end

using Random: randperm
function learn2classify_asgd_1layer(
        f_a::Function, 
        df_a::Function, 
        grad_loss::Function,
        x::Matrix, 
        y::Matrix, 
        W0::Matrix, 
        b0::Vector,
        mu::Number=1e-3, 
        iters::Integer=500, 
        batch_size::Integer=10
    )

    d = size(W0, 2) #number of inputs
    n = size(W0, 1) # number of neurons
    N = size(x, 2) # number of training samples
 
    W = W0
    b = b0
    
    loss = zeros(iters)

    lambdak = 0
    qk = W
    pk = b
    for i in 1:iters
        batch_idx = randperm(N)
        batch_idx = batch_idx[1:min(batch_size, N)]
        
        dW, db, loss_i = grad_loss(f_a, df_a, x[:, batch_idx], y[:, batch_idx], W, b)
        
        qkp1 = W - mu * dW
        pkp1 = b - mu * db

        lambdakp1 = (1 + sqrt(1 + 4 * lambdak^2)) / 2
        gammak = (1 - lambdak) / lambdakp1

        W = (1 - gammak) * qkp1 + gammak * qk
        b = (1 - gammak) * pkp1 + gammak * pk

        qk = qkp1
        pk = pkp1
        lambdak = lambdakp1

        loss[i] = loss_i
    end
    return W, b, loss
end


function load_digit_data(digit::Integer, nx::Integer=28, ny::Integer=28, nrep::Integer=1000)
    filepath = "data" * string(digit)
    
    x = open(filepath, "r") do file
        reshape(read(file), (nx, ny, nrep)) 
    end
    
   return x
end
