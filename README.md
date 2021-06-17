# MtSpMV.jl

[![Build Status](https://travis-ci.com/jkrch/MtSpMV.jl.svg?branch=master)](https://travis-ci.com/jkrch/MtSpMV.jl)

This package provides multithreaded sparse matrix-vector multiplication (SpMV) for the CSR format and is written in pure Julia.

## Usage

By `using MtSpMV` all `mul!` (and *) operations will run on multiple threads if the input matrix is of type `Transpose{<:Any, <:SparseMatrixCSC}` or `Adjoint{<:Any, <:SparseMatrixCSC}` and the input vector is of type `StridedVector`.

## Example

```julia
using SparseArrays, MtSpMV

n = 10000000
A = sprand(n, n, 5/n)
x = rand(n)
transA = transpose(sparse(transpose(A)))
y = zeros(n)

mul!(y, transA, x)
```
