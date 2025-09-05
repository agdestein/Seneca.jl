if false
    include("src/Seneca.jl")
    using .Seneca
    import .Seneca as S
end

using LinearAlgebra
using Random
import Seneca as S
using WGLMakie
using CUDA
lines([1,2,3])

g = S.Grid{2}(; l = 1.0, n = 1024, backend = CUDABackend())

cache = S.getcache(g);
ustart = S.randomfield(g; kpeak = 5);
# ustart = S.taylorgreen(g, cache.plan);
energy = round(S.energy(ustart); sigdigits = 4)

# Burn-in
# u = map(copy, ustart);
# u = map(zero, ustart);
# u = ustart;
obs = let
    o = Observable(u)
    temp = cache.vi_vj |> Array
    vort = map(o) do u
        S.z_vort!(cache.vi_vj, cache.du.x, u, cache.plan, g)
        copyto!(temp, cache.vi_vj)
        temp
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    range = logrange(1e-1, 5e2, 100)
    # range = logrange(1e-3, 5e0, 50)
    sl = Makie.Slider(fig[0, 1]; range, startvalue = 1, update_while_dragging=true)
    colorrange = map(sl.value) do sl
        (-sl, sl)
    end
    hm = image!(ax, vort; colormap = :seaborn_icefire_gradient, colorrange)
    Colorbar(fig[1, 2], hm)
    display(fig)
    o
end;

visc = 5e-6
let
    t = 0.0
    cfl = 0.35
    tstop = 1e0
    Δt_set = 3e-3
    ou = S.ouforcer(g, 2.3)
    t_ou, estar = 0.005, 0.01
    σ = sqrt(estar / t_ou)
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * S.propose_timestep(u, cache, visc, g)
        Δt = min(Δt, tstop - t)
        S.wray3!(u, cache, Δt, visc, g)
        # S.forwardeuler!(u, cache, Δt, visc, g)
        # Δt = Δt_set; S.abcn!(u, cache, Δt, visc, g; firststep = i == 1)
        t += Δt
        randn!(ou.b)
        @. ou.b *= sqrt(2 * σ^2 * Δt / t_ou)
        @. ou.b += (1 - Δt / t_ou) * ou.bold
        copyto!(ou.bold, ou.b)
        for j = 1:S.dim(g)
            @. u[j][ou.iuse] += Δt * ou.b[:, j]
        end
        S.apply!(S.project!, g, (u, g))
        if i % 10 == 0
            energy = S.energy(u)
            @info join(
                [
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(energy; sigdigits = 4))",
                ],
                ",\t",
            )
        end
        if i % 200 == 0
            obs[] = u
            sleep(0.01)
        end
    end
    t
end

let
    stat = S.turbulence_statistics(u, visc, g)
    s = S.spectrum(u, g)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    k = [2, 500]
    kolmo = @. 2e0 * stat.diss^(1 / 3) * k^(-3)
    kscale = stat.l_kol
    escale = stat.diss^(-2 / 3) * stat.l_kol^(-3)
    lines!(ax, kscale * s.k, escale * s.s)
    lines!(kscale * k, escale * kolmo)
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
