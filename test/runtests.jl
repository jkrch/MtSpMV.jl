using Test
using SparseArrays
using LinearAlgebra
using MtSpMV


# Test parallel SpMV
function spmv(m, n)

    # Create random test matrix and random test vector
    A = sprand(m, n, 5/m)
    transA = transpose(sparse(transpose(A)))
    x = rand(n)
    
    # Compute SpMV in serial for CSC matrix
    y_ser = zeros(m)
    for i = 1:n
        for j = A.colptr[i]:(A.colptr[i+1] - 1)
            y_ser[A.rowval[j]] += A.nzval[j] * x[i]
        end
    end
    
    # Compute SpMV in parallel with mul! and test
    y_par = zeros(m)
    mul!(y_par, transA, x)
    @test y_ser == y_par
    
    # Compute SpMV in parallel with * and test
    @test y_ser == transA * x
end


# Tests for quadratic and nonquadratic matrices
@testset "spmv" begin
    
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
