# Multithreaded matrix-vector multiplication for 'Transpose{<:Any,<:SparseMatrixCSC}' 
# using Threads.@threads
# Why the unused α and β? See: https://github.com/JuliaLang/julia/blob/381693d3dfc9b7072707f6d544f82f6637fc5e7c/stdlib/LinearAlgebra/src/matmul.jl#L202   
function mul!(y::StridedVector, transA::Transpose{<:Any, <:SparseMatrixCSC}, x::StridedVector, α::Number, β::Number) 
    size(transA)[2] == length(x) || throw(DimensionMismatch("transA.n != length(x)"))
    A = transA.parent  
    nzval = A.nzval
    colval = A.rowval
    rowptr = A.colptr
    m = A.n
    @inbounds Threads.@threads for i = 1:m
        y[i] = 0 
        @inbounds for j = rowptr[i]:(rowptr[i + 1] - 1)
            y[i] += nzval[j] * x[colval[j]]
        end
    end
    y
end
*(transA::Transpose{<:Any, <:SparseMatrixCSC}, x::StridedVector{Tx}) where {Tx} =
    (T = promote_op(matprod, eltype(transA), Tx); mul!(similar(x, T, size(transA, 1)), transA, x, true, false))


# Multithreaded matrix-vector multiplication for 'Adjoint{<:Any,<:SparseMatrixCSC}' 
# using Threads.@threads
# Why the unused α and β? See: https://github.com/JuliaLang/julia/blob/381693d3dfc9b7072707f6d544f82f6637fc5e7c/stdlib/LinearAlgebra/src/matmul.jl#L202   
function mul!(y::StridedVector, adjA::Adjoint{<:Any, <:SparseMatrixCSC}, x::StridedVector, α::Number, β::Number) 
    size(adjA)[2] == length(x) || throw(DimensionMismatch("adjA.n != length(x)"))
    A = adjA.parent  
    nzval = A.nzval
    colval = A.rowval
    rowptr = A.colptr
    m = A.n
    @inbounds Threads.@threads for i = 1:m
        y[i] = 0 
        @inbounds for j = rowptr[i]:(rowptr[i + 1] - 1)
            y[i] += nzval[j] * x[colval[j]]
        end
    end
    y
end
*(transA::Adjoint{<:Any, <:SparseMatrixCSC}, x::StridedVector{Tx}) where {Tx} =
    (T = promote_op(matprod, eltype(transA), Tx); mul!(similar(x, T, size(transA, 1)), adjA, x, true, false))

