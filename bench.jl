
if false
    include("src/Seneca.jl")
    using .Seneca
    import .Seneca as S
end

using BenchmarkTools
using LinearAlgebra
using Random
using Seneca
using WGLMakie
using CUDA
lines([1, 2, 3])

g = Grid{3}(;
    l = 1.0,
    n = 512,
    backend = CUDABackend(),
)

cache = getcache(g);
ustart = randomfield(g; kpeak = 5);
e = round(energy(ustart); sigdigits = 4)

# u = map(copy, ustart)
u = ustart

visc = 3e-4
cfl = 0.85
Δt = cfl * propose_timestep(u, g, visc, cache)
@show Δt
@benchmark wray3!(convectiondiffusion!, u, Δt, g, cache; visc)


# 1 thread CPU
# Δt = 0.0008584334033841037
# BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
#  Range (min … max):  3.564 s …  3.568 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     3.566 s             ┊ GC (median):    0.00%
#  Time  (mean ± σ):   3.566 s ± 3.333 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

# 32 threads CPU
# BenchmarkTools.Trial: 2 samples with 1 evaluation per sample.
#  Range (min … max):  3.334 s …   3.403 s  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     3.369 s              ┊ GC (median):    0.00%
#  Time  (mean ± σ):   3.369 s ± 48.980 ms  ┊ GC (mean ± σ):  0.00% ± 0.00%

# Δt = 0.000786996520185516
#
# BenchmarkTools.Trial: 57 samples with 1 evaluation per sample.
#  Range (min … max):  87.707 ms …  88.879 ms  ┊ GC (min … max): 0.00% … 0.00%
#  Time  (median):     88.039 ms               ┊ GC (median):    0.00%
#  Time  (mean ± σ):   88.084 ms ± 281.648 μs  ┊ GC (mean ± σ):  0.00% ± 0.00%

512
# CPU
# Single result which took 35.457 s (0.00% GC) to evaluate,
#
# GPU
# Range (min … max):  744.823 ms … 745.132 ms  ┊ GC (min … max): 2.01% … 2.07%
# Time  (median):     744.996 ms               ┊ GC (median):    2.02%
# Time  (mean ± σ):   744.976 ms ± 116.649 μs  ┊ GC (mean ± σ):  2.02% ± 0.02%
