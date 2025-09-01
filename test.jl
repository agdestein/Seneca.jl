if false
    include("src/Seneca.jl")
    using .Seneca
end

using LinearAlgebra
using Random
import Seneca as S
using WGLMakie
using CUDA

g = S.Grid(; l = 1.0, n = 32)
g = S.Grid(; l = 1.0, n = 256, backend = CUDABackend())

cache = S.getcache(g);
ustart = S.randomfield(g; kpeak = 5);
# ustart = S.taylorgreen(g, cache.plan);
# u = map(copy, ustart);
u = ustart;
energy = round(sum(u -> sum(abs2, u) + sum(abs2, u[2:end-1, :, :]), u); sigdigits = 4)
# u.x ./= sqrt(energy);
# u.y ./= sqrt(energy);
# u.z ./= sqrt(energy);

# Burn-in
# u = map(copy, ustart);
obs = let
    o = Observable(u)
    vort = map(o) do u
        S.z_vort!(cache.vi_vj, cache.du.x, u, cache.plan, g)
        view(cache.vi_vj, :, :, g.n) |> Array
    end
    fig, ax, hm = heatmap(vort; colormap = :seaborn_icefire_gradient, colorrange = (-100, 100))
    Colorbar(fig[1, 2], hm)
    display(fig)
    o
end
let
    t = 0.0
    cfl = 0.15
    tstop = 1e0
    Δt_set = 3e-4
    visc = 5e-4
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * S.propose_timestep(u, cache, visc, g)
        # Δt = Δt_set
        Δt = min(Δt, tstop - t)
        S.wray3!(u, cache, Δt, visc, g)
        # S.forwardeuler!(u, cache, Δt, visc, g)
        # S.abcn!(u, cache, Δt, visc, g; firststep = i == 1)
        t += Δt
        @info join(
            [
                "t = $(round(t; sigdigits = 4))",
                "Δt = $(round(Δt; sigdigits = 4))",
                # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                "energy = $(round(sum(u -> sum(abs2, u), u); sigdigits = 4))",
            ],
            ",\t",
        )
        if i % 1 == 0
            obs[] = u
            sleep(0.01)
        end
    end
    t
end

let
    s = S.spectrum(u, g)
    fig = lines(s.k, s.s, axis = (; xscale = log10, yscale = log10))
    k = [10, 100]
    lines!(k, k .^ (-5 / 3))
    # ylims!(1e-7, 1)
    fig
end

let
    a = S.scalarfield(g)
    b = S.scalarfield(g)
    randn!(b)
    S.apply!(S.twothirds!, g, a, b, g)
    a[:, :, 1] .|> abs |> Array |> heatmap
end
