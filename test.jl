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

g = S.Grid(; l = 1.0, n = 32)
g = S.Grid(; l = 1.0, n = 256, backend = CUDABackend())

cache = S.getcache(g);
# ustart = S.randomfield(g; kpeak = 5);
ustart = S.taylorgreen(g, cache.plan);
energy = round(sum(u -> sum(abs2, u) + sum(abs2, u[2:(end-1), :, :]), ustart); sigdigits = 4)

# Burn-in
u = map(copy, ustart);
# u = ustart;
obs = let
    o = Observable(u)
    vort = map(o) do u
        S.z_vort!(cache.vi_vj, cache.du.x, u, cache.plan, g)
        # copyto!(cache.du.x, u.x)
        # cache.du.x .*= g.n^3
        # ldiv!(cache.vi_vj, cache.plan, cache.du.x)
        view(cache.vi_vj, :, :, g.n ÷ 4) |> Array
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    range = logrange(1e-1, 5e2, 100)
    # range = logrange(1e-3, 5e0, 50)
    sl = Makie.Slider(fig[0, 1]; range, startvalue = 1, update_while_dragging=true)
    colorrange = map(sl.value) do sl
        (-sl, sl)
    end
    amp = 400
    hm = image!(ax, vort; colormap = :seaborn_icefire_gradient, colorrange)
    Colorbar(fig[1, 2], hm)
    display(fig)
    o
end;

visc = 4e-4
let
    t = 0.0
    cfl = 0.35
    tstop = 5e0
    Δt_set = 3e-3
    ou = S.ouforcer(g, 1.5)
    t_ou, estar = 0.01, 0.1
    σ = sqrt(estar / t_ou)
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * S.propose_timestep(u, cache, visc, g)
        # Δt = min(Δt, tstop - t)
        # S.wray3!(u, cache, Δt, visc, g)
        # S.forwardeuler!(u, cache, Δt, visc, g)
        Δt = Δt_set; S.abcn!(u, cache, Δt, visc, g; firststep = i == 1)
        t += Δt
        # randn!(ou.b)
        # @. ou.b *= sqrt(2 * σ^2 * Δt / t_ou)
        # @. ou.b += (1 - Δt / t_ou) * ou.bold
        # copyto!(ou.bold, ou.b)
        # @. u.x[ou.iuse] += Δt * ou.b[:, 1]
        # @. u.y[ou.iuse] += Δt * ou.b[:, 2]
        # @. u.z[ou.iuse] += Δt * ou.b[:, 3]
        # S.apply!(S.project!, g, u, g)
        energy = sum(u -> sum(abs2, u) + sum(abs2, view(u,(2:(size(u, 1)-1)),:,:)), u) / 2
        @info join(
            [
                "t = $(round(t; sigdigits = 4))",
                "Δt = $(round(Δt; sigdigits = 4))",
                # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                "energy = $(round(energy; sigdigits = 4))",
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
    stat = S.turbulence_statistics(u, visc, g)
    s = S.spectrum(u, g)
    fig = Figure()
    ax = Axis(fig[1, 1]; xscale = log10, yscale = log10)
    k = [3, 60]
    kolmo = @. 0.5 * stat.diss^(2 / 3) * k^(-5 / 3)
    kscale = stat.l_kol
    escale = stat.diss^(-2 / 3) * stat.l_kol^(-5 / 3)
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
