using LinearAlgebra
using SparseArrays


# Test parallel csr matrix-vector product
function spmv(m, n)

    # Create random test matrix and vector
    A = sprand(m, n, 5/m)
    transA = transpose(sparse(transpose(A)))
    x = rand(n)
    y = A * x

    # Test empty result
    @test y == mul!(zeros(m), transA, x)
    
    # Test nonempty result
    @test y == mul!(rand(m), transA, x)
    
    # Test with result initialization
    @test y == transA * x
end


# Tests for quadratic and nonquadratic matrices
@testset "mul" begin
    
    @testset "quadratic matrix" begin
        spmv(100, 100)
        spmv(1000, 1000)
    end
    
    @testset "nonquadratic matrix, nrows < ncols" begin
        spmv(80, 100)
        spmv(800, 1000)
    end
    
    @testset "nonquadratic matrix, nrows > ncols" begin
        spmv(100, 80)
        spmv(1000, 800)
    end
end
