# Load EarthSciMLBase's BlockDiagonal machinery (the block-diagonal Jacobian
# prototype the user asked for) WITHOUT dragging in the whole EarthSciMLBase
# dependency tree (ModelingToolkit/Catalyst). `map_algorithm.jl` supplies the
# MapAlgorithm dispatch `blockdiagonal.jl` needs; it references AcceleratedKernels
# (`AK`) — imported here — for the MapKernel/mapreduce_range paths (BlockDiagonal's
# `ldiv!` hardcodes `MapKernel()`).
module BlockDiag

import AcceleratedKernels as AK          # map_algorithm.jl's `AK` (ldiv!/lufact paths)

const _MB_SRC = "/Users/ctessum/code/earthsciml/EarthSciMLBase/src"
include(joinpath(_MB_SRC, "map_algorithm.jl"))     # MapAlgorithm, MapBroadcast, map_closure_to_range, mapreduce_range
include(joinpath(_MB_SRC, "blockdiagonal.jl"))     # BlockDiagonal + LU/ldiv!/mul!/+I, LinearSolve glue

export BlockDiagonal, MapBroadcast, block, nblocks

end # module
