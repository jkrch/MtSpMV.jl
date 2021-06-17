"""
Multithreaded CSR Matrix-Vector Product
"""
module MtSpMV

using SparseArrays, LinearAlgebra

using Base: promote_op

import Base: *
import LinearAlgebra: mul!, matprod

export mul!, *

include("parallellinalg.jl")

end
