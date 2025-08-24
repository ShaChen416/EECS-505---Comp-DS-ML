using SparseArrays, LinearAlgebra
function first_diffs_2d_matrix(m, n)
#
# Syntax:       A = first_diffs_2d_matrix_sol(m, n)
#               
# Inputs:       m and n are positive integers
#               
# Outputs:      A is a 2mn x mn sparse matrix such that A * X[:] computes the
#               first differences down the columns (along x direction)
#               and across the (along y direction) of the m x n matrix X
#
    
# Hint: You will need Dn and Dm 
     Dn = spdiagm(0 => -ones(n), 1 => ones(n-1))
     Dn[n, 1] = 1    
    
     Dm = spdiagm(0 => -ones(m), 1 => ones(m-1))
     Dm[m, 1] = 1    
      
     In = spdiagm(0 => ones(n))
     Im = spdiagm(0 => ones(m))
    
     A = vcat(kron(In,Dm),kron(Dn,Im) )
   
    return A
    
end
