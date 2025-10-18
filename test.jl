if false
    include("src/Seneca.jl")
    using .Seneca
    import .Seneca as S
end

using LinearAlgebra
using Random
using Seneca
using WGLMakie
using CUDA
lines([1, 2, 3])

g = Grid{3}(; l = 1.0, n = 256, backend = CUDABackend())

cache = getcache(g);
ustart = randomfield(g; kpeak = 5);
# ustart = taylorgreen(g, cache.plan);
e = round(energy(ustart); sigdigits = 4)

# Burn-in
u = map(copy, ustart);
# u = map(zero, ustart);
# u = ustart;

obs = let
    o = Observable(u)
    temp = cache.vi_vj |> Array
    vort = map(o) do u
        z_vort!(cache.vi_vj, cache.du.x, u, cache.plan, g)
        copyto!(temp, cache.vi_vj)
        view(temp,:,:,1)
    end
    fig = Figure()
    ax = Axis(fig[1, 1])
    range = logrange(1e-1, 5e2, 100)
    # range = logrange(1e-3, 5e0, 50)
    sl = Makie.Slider(fig[0, 1]; range, startvalue = 50, update_while_dragging = true)
    colorrange = map(sl.value) do sl
        (-sl, sl)
    end
    hm = image!(
        ax,
        vort;
        colormap = :seaborn_icefire_gradient,
        colorrange,
        interpolate = false,
    )
    Colorbar(fig[1, 2], hm)
    display(fig)
    o
end;

visc = 3e-4
let
    t = 0.0
    cfl = 0.85
    tstop = 1e0
    Δt_set = 3e-3
    ou = ouforcer(g, 2.3)
    t_ou, estar = 0.005, 0.01
    # ou = S.ouforcer(g, 1.5)
    # t_ou, estar = 0.01, 0.1
    σ = sqrt(estar / t_ou)
    i = 0
    while t < tstop
        i += 1
        Δt = cfl * propose_timestep(u, g, visc, cache)
        Δt = min(Δt, tstop - t)
        wray3!(convectiondiffusion!, u, Δt, g, cache; visc)
        # forwardeuler!(u, cache, Δt, visc, g)
        # Δt = Δt_set; abcn!(u, cache, Δt, visc, g; firststep = i == 1)
        t += Δt
        randn!(ou.b)
        @. ou.b *= sqrt(2 * σ^2 * Δt / t_ou)
        @. ou.b += (1 - Δt / t_ou) * ou.bold
        copyto!(ou.bold, ou.b)
        for j = 1:dim(g)
            @. u[j][ou.iuse] += Δt * ou.b[:, j]
        end
        apply!(project!, g, (u, g))
        if i % 1 == 0
            foreach(u -> apply!(twothirds!, g, (u, g)), u) # Ensure 2/3 dealiasing
            e = energy(u)
            @info join(
                [
                    "t = $(round(t; sigdigits = 4))",
                    "Δt = $(round(Δt; sigdigits = 4))",
                    # "umax = $(round(maximum(u -> maximum(abs, u), u); sigdigits = 4))",
                    "energy = $(round(e; sigdigits = 4))",
                ],
                ",\t",
            )
        end
        if i % 1 == 0
            obs[] = u
            sleep(0.01)
        end
    end
    t
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
    s = turbulence_statistics(u, visc, g)
    s |> pairs
end

let
    a = S.scalarfield(g)
    b = S.scalarfield(g)
    randn!(b)
    S.apply!(S.twothirds!, g, a, b, g)
    a[:, :, 1] .|> abs |> Array |> heatmap
end
