if false
    include("src/Seneca.jl")
    using .Seneca
    import .Seneca as S
end

using JLD2
using LinearAlgebra
using Random
using Seneca
using WGLMakie
using CUDA
lines([1, 2, 3])

outdir = joinpath(@__DIR__, "output") |> mkpath

g = Grid{3}(; l = 2π, n = 512, backend = CUDABackend())
visc = 2e-4

cache = getcache(g);
u = taylorgreen(g, cache.plan);
e = round(energy(u); sigdigits = 4)

times, dissipation = let
    t = 0.0
    cfl = 0.85
    tstop = 2e1
    Δt = 0.0
    i = 0
    times = zeros(0)
    dissipation = zeros(0)
    S = tensorfield(g)
    while t < tstop
        if i > 0 # Skip first step to get initial statistics
            Δt = cfl * propose_timestep(u, g, visc, cache)
            Δt = min(Δt, tstop - t)
            wray3!(convectiondiffusion!, u, Δt, g, cache; visc)
            t += Δt
        end
        if i % 1 == 0
            foreach(u -> apply!(twothirds!, g, (u, g)), u) # Remove polluted components
            apply!(strainrate!, g, (S, u, g))
            d =
                2 *
                visc *
                (
                    getenergy(S.xx) +
                    getenergy(S.yy) +
                    getenergy(S.zz) +
                    2 * getenergy(S.xy) +
                    2 * getenergy(S.yz) +
                    2 * getenergy(S.zx)
                )
            push!(times, t)
            push!(dissipation, d)
            @info join(
                [
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    # "energy = $(round(e; sigdigits = 4))",
                    "dissipation = $(round(d; sigdigits = 4))",
                ],
                ",\t",
            )
        end
        i += 1
    end
    times, dissipation
end

jldsave("$outdir/taylorgreen-dissipation.jld2"; times, dissipation)
times, dissipation = load("$outdir/taylorgreen-dissipation.jld2", "times", "dissipation")

let
    fig = Figure()
    ax = Axis(fig[1, 1]; xlabel = "time", ylabel = "dissipation")
    lines!(ax, times, dissipation)
    fig
end

let
    D = dim(g)
    stat = turbulence_statistics(u, visc, g)
    s = spectrum(u, g)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    kmax = div(g.n, 2)
    kcut = div(2 * kmax, 3)
    # k = [2, 500]
    k = [2, g.n / 8]
    if D == 2
        kolmo = @. 2e0 * stat.diss^(1 / 3) * k^(-3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    elseif D == 3
        kolmo = @. 5e-1 * stat.diss^(2 / 3) * k^(-5 / 3)
        escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
    end
    kscale = stat.l_kol
    lines!(ax, kscale * s.k, escale * s.s)
    lines!(kscale * k, escale * kolmo)
    # vlines!(kscale * kcut)
    # ylims!(1e-7, 1)
    fig
end

let
    s = S.turbulence_statistics(u, visc, g)
    s |> pairs
end

let
    a = S.scalarfield(g)
    b = S.scalarfield(g)
    randn!(b)
    S.apply!(S.twothirds!, g, a, b, g)
    a[:, :, 1] .|> abs |> Array |> heatmap
end
