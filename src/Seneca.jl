"Pseudo-spectral solver for the 3D incompressible Navier-Stokes equations."
module Seneca

using AbstractFFTs
using Adapt
using FFTW
using LinearAlgebra
using KernelAbstractions
using Random

struct Grid{D,T,B}
    "Domain side length."
    l::T

    "Number of grid points in each dimension."
    n::Int

    """
    KernelAbstractions.jl hardware backend.
    For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
    """
    backend::B

    "Kernel work group size."
    workgroupsize::Int

    Grid{D}(; l, n, backend = CPU(), workgroupsize = 64) where {D} =
        new{D,typeof(l),typeof(backend)}(l, n, backend, workgroupsize)
end

@inline dim(::Grid{D}) where {D} = D
@inline tensordim(::Grid{2}) = 3
@inline tensordim(::Grid{3}) = 6

# Like fftfreq, but with proper type
@inline fftfreq_int(g::Grid, i::Int) = i - 1 - ifelse(i <= (g.n + 1) >> 1, 0, g.n)
@inline fftfreq_int(g::Grid, I::CartesianIndex{2}) = I.I[1] - 1, fftfreq_int(g, I.I[2])
@inline fftfreq_int(g::Grid, I::CartesianIndex{3}) =
    I.I[1] - 1, fftfreq_int(g, I.I[2]), fftfreq_int(g, I.I[3])
@inline wavenumbers(g::Grid, I::CartesianIndex{2}) =
    pi / g.l * 2 * (I.I[1] - 1), pi / g.l * 2 * fftfreq_int(g, I.I[2])
@inline wavenumbers(g::Grid, I::CartesianIndex{3}) = (
    pi / g.l * 2 * (I.I[1] - 1),
    pi / g.l * 2 * fftfreq_int(g, I.I[2]),
    pi / g.l * 2 * fftfreq_int(g, I.I[3]),
)
@inline function squared_wavenumber(g::Grid{2}, I)
    kx, ky = fftfreq_int(g, I)
    kx^2 + ky^2
end
@inline function squared_wavenumber(g::Grid{3}, I)
    kx, ky, kz = fftfreq_int(g, I)
    kx^2 + ky^2 + kz^2
end

ndrange((; n)::Grid{2}) = div(n, 2) + 1, n
ndrange((; n)::Grid{3}) = div(n, 2) + 1, n, n

function apply!(kernel!, grid, args; ndrange = ndrange(grid))
    (; backend, workgroupsize) = grid
    kernel!(backend, workgroupsize)(args...; ndrange)
    KernelAbstractions.synchronize(backend)
    nothing
end

scalarfield(g::Grid{D,T}) where {D,T} =
    KernelAbstractions.zeros(g.backend, Complex{T}, ndrange(g))
vectorfield(g::Grid{2}) = (; x = scalarfield(g), y = scalarfield(g))
vectorfield(g::Grid{3}) = (; x = scalarfield(g), y = scalarfield(g), z = scalarfield(g))
tensorfield(g::Grid{2}) = (; xx = scalarfield(g), yy = scalarfield(g), xy = scalarfield(g))
tensorfield(g::Grid{3}) = (;
    xx = scalarfield(g),
    yy = scalarfield(g),
    zz = scalarfield(g),
    xy = scalarfield(g),
    yz = scalarfield(g),
    zx = scalarfield(g),
)

spacescalarfield(g::Grid{D,T}) where {D,T} =
    KernelAbstractions.zeros(g.backend, T, ntuple(Returns(g.n), D))
spacevectorfield(g::Grid{2}) = (; x = spacescalarfield(g), y = spacescalarfield(g))
spacevectorfield(g::Grid{3}) =
    (; x = spacescalarfield(g), y = spacescalarfield(g), z = spacescalarfield(g))

@kernel function project!(u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumbers(g, I)
    ux, uy = u.x[I], u.y[I]
    p = (kx * ux + ky * uy) / (kx * kx + ky * ky)
    p = ifelse(I.I == (1, 1), zero(p), p) # Leave constant mode intact
    u.x[I] = ux - kx * p
    u.y[I] = uy - ky * p
end
@kernel function project!(u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    p = (kx * ux + ky * uy + kz * uz) / (kx * kx + ky * ky + kz * kz)
    p = ifelse(I.I == (1, 1, 1), zero(p), p) # Leave constant mode intact
    u.x[I] = ux - kx * p
    u.y[I] = uy - ky * p
    u.z[I] = uz - kz * p
end

@kernel function twothirds!(ubar, u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = fftfreq_int(g, I)
    K = div(g.n, 2)
    ix = 3 * abs(kx) ≤ 2 * K
    iy = 3 * abs(ky) ≤ 2 * K
    ubar[I] = ifelse(ix & iy, u[I], zero(eltype(u)))
    # ubar[I] = ifelse(9 * (kx^2 + ky^2) ≤ 4 * K^2, u[I], zero(eltype(u)))
end
@kernel function twothirds!(ubar, u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = fftfreq_int(g, I)
    K = div(g.n, 2)
    ix = 3 * abs(kx) ≤ 2 * K
    iy = 3 * abs(ky) ≤ 2 * K
    iz = 3 * abs(kz) ≤ 2 * K
    ubar[I] = ifelse(ix & iy & iz, u[I], zero(eltype(u)))
    # ubar[I] = ifelse(9 * (kx^2 + ky^2 + kz^2) ≤ 4 * K^2, u[I], zero(eltype(u)))
end

getplan(grid) = plan_rfft(spacescalarfield(grid))

function nonlinearity!(σ, vi_vj, v, u, plan, g::Grid{2})
    temp = σ.xx # Use σ.xx as temporary complex storage
    fac = g.n^2
    for i = 1:2
        apply!(twothirds!, g, (temp, u[i], g)) # Zero out high wavenumbers
        ldiv!(v[i], plan, temp) # Inverse transform
        v[i] .*= fac # FFT factor
    end
    #! format: off
    @. vi_vj = v.x * v.x; mul!(σ.xx, plan, vi_vj); (σ.xx ./= fac)
    @. vi_vj = v.y * v.y; mul!(σ.yy, plan, vi_vj); (σ.yy ./= fac)
    @. vi_vj = v.x * v.y; mul!(σ.xy, plan, vi_vj); (σ.xy ./= fac)
    #! format: on
    nothing
end
function nonlinearity!(σ, vi_vj, v, u, plan, g::Grid{3})
    temp = σ.xx # Use σ.xx as temporary complex storage
    fac = g.n^3
    for i = 1:3
        apply!(twothirds!, g, (temp, u[i], g)) # Zero out high wavenumbers
        ldiv!(v[i], plan, temp) # Inverse transform
        v[i] .*= fac # FFT factor
    end
    #! format: off
    @. vi_vj = v.x * v.x; mul!(σ.xx, plan, vi_vj); (σ.xx ./= fac)
    @. vi_vj = v.y * v.y; mul!(σ.yy, plan, vi_vj); (σ.yy ./= fac)
    @. vi_vj = v.z * v.z; mul!(σ.zz, plan, vi_vj); (σ.zz ./= fac)
    @. vi_vj = v.x * v.y; mul!(σ.xy, plan, vi_vj); (σ.xy ./= fac)
    @. vi_vj = v.y * v.z; mul!(σ.yz, plan, vi_vj); (σ.yz ./= fac)
    @. vi_vj = v.z * v.x; mul!(σ.zx, plan, vi_vj); (σ.zx ./= fac)
    #! format: on
    nothing
end

@kernel function viscosity!(σ, u, visc, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumbers(g, I)
    ux, uy = u.x[I], u.y[I]
    σ.xx[I] -= visc * im * kx * ux
    σ.yy[I] -= visc * im * ky * uy
    σ.xy[I] -= visc * im * (ky * ux + kx * uy)
end
@kernel function viscosity!(σ, u, visc, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    σ.xx[I] -= visc * im * kx * ux
    σ.yy[I] -= visc * im * ky * uy
    σ.zz[I] -= visc * im * kz * uz
    σ.xy[I] -= visc * im * (ky * ux + kx * uy)
    σ.yz[I] -= visc * im * (kz * uy + ky * uz)
    σ.zx[I] -= visc * im * (kx * uz + kz * ux)
end

function stress!(σ, vi_vj, v, u, plan, visc, g::Grid)
    # foreach(s -> fill!(s, 0), σ)
    nonlinearity!(σ, vi_vj, v, u, plan, g)
    apply!(viscosity!, g, (σ, u, visc, g))
end

@kernel function vectordivergence!(div, u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumbers(g, I)
    div[I] = im * kx * u.x[I] + im * ky * u.y[I]
end
@kernel function vectordivergence!(div, u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(g, I)
    div[I] = im * kx * u.x[I] + im * ky * u.y[I] + im * kz * u.z[I]
end

@kernel function tensordivergence!(div, σ, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumbers(g, I)
    div.x[I] = -im * kx * σ.xx[I] - im * ky * σ.xy[I]
    div.y[I] = -im * kx * σ.xy[I] - im * ky * σ.yy[I]
end
@kernel function tensordivergence!(div, σ, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(g, I)
    div.x[I] = -im * kx * σ.xx[I] - im * ky * σ.xy[I] - im * kz * σ.zx[I]
    div.y[I] = -im * kx * σ.xy[I] - im * ky * σ.yy[I] - im * kz * σ.yz[I]
    div.z[I] = -im * kx * σ.zx[I] - im * ky * σ.yz[I] - im * kz * σ.zz[I]
end

profile(k, kpeak) = k^4 * exp(-2 * (k / kpeak)^2)

"Taylor-Green vortex."
function taylorgreen(g::Grid{2}, plan)
    (; l, n, backend) = g
    h = l / n
    x = range(h / 2, 1 - h / 2, n) |> Array |> adapt(backend)
    y = reshape(x, 1, :)
    v = spacescalarfield(g)
    fac = n^2
    #! format: off
    @. v =  sinpi(2x / l) * cospi(2y / l); ux = plan * v; ux ./= fac
    @. v = -cospi(2x / l) * sinpi(2y / l); uy = plan * v; uy ./= fac
    #! format: on
    v = nothing
    u = (; x = ux, y = uy)
    apply!(project!, g, (u, g))
    u
end
function taylorgreen(g::Grid{3}, plan)
    (; l, n, backend) = g
    h = l / n
    x = range(h / 2, 1 - h / 2, n) |> Array |> adapt(backend)
    y = reshape(x, 1, :)
    z = reshape(x, 1, 1, :)
    v = spacescalarfield(g)
    fac = n^3
    #! format: off
    @. v =  sinpi(2x / l) * cospi(2y / l) * sinpi(2z / l) / 2; ux = plan * v; ux ./= fac
    @. v = -cospi(2x / l) * sinpi(2y / l) * sinpi(2z / l) / 2; uy = plan * v; uy ./= fac
    #! format: on
    v = nothing
    uz = zero(ux)
    u = (; x = ux, y = uy, z = uz)
    apply!(project!, g, (u, g))
    u
end

"""
Make random velocity field with prescribed energy spectrum profile.

Note: The profile takes the scalar wavenumber norm as input,
define it as `profile(k)`.
"""
function randomfield(grid; kpeak = 5, totalenergy = 1, rng = Random.default_rng())
    # Mask for active wavenumbers: kleft ≤ k < kleft + 1
    # Do everything squared to avoid floats
    @kernel function mask!(mask, kleft, g)
        I = @index(Global, Cartesian)
        k2 = squared_wavenumber(g, I)
        mask[I] = kleft^2 ≤ k2 < (kleft + 1)^2
    end

    # Create random field and make it divergence free
    u = vectorfield(grid)
    foreach(u -> randn!(rng, u), u)
    apply!(project!, grid, (u, grid))

    # RFFT exploits conjugate symmetry, so we only need half the modes
    kmax = div(grid.n, 2)
    range = ndrange(grid)

    # Allocate arrays
    E = similar(u.x, range...)
    Emask = similar(E)
    mask = similar(E, Bool)

    # Compute energy
    if dim(grid) == 2
        @. E = (abs2(u.x) + abs2(u.y)) / 2
    else
        @. E = (abs2(u.x) + abs2(u.y) + abs2(u.z)) / 2
    end

    # Maximum partially resolved wavenumber (sqrt(dim) * kmax)
    kdiag = floor(Int, sqrt(3) * div(grid.n, 2))

    # Sum of shell weights 
    totalprofile = sum(k -> profile(k, kpeak), 0:kdiag)

    # Adjust energy in each partially resolved shell [k, k+1)
    for k = 0:kdiag
        apply!(mask!, grid, (mask, k, grid)) # Shell mask
        @. Emask = mask * E
        Eshell = sum(Emask) + sum(view(Emask, 2:kmax, :, :)) # Current energy in shell
        E0 = totalenergy * profile(k, kpeak) / totalprofile # Desired energy in shell
        factor = sqrt(E0 / Eshell) # E = u^2 / 2
        for i = 1:dim(grid)
            ui = u[i]
            @. ui = ifelse(mask, factor * ui, ui)
        end
    end

    # The velocity now has
    # the correct spectrum,
    # random phase shifts,
    # random orientations, 
    # and is also divergence free.
    u
end

function energy(u)
    kmax = size(u.x, 1) - 1
    sum(u -> sum(abs2, u) + sum(abs2, selectdim(u, 1, 2:kmax)), u) / 2
end

@kernel function z_vort_kernel!(vort, u, grid)
    I = @index(Global, Cartesian)
    k = wavenumbers(grid, I)
    kx, ky = k[1], k[2]
    ux, uy = u.x[I], u.y[I]
    vort[I] = -im * ky * ux + im * kx * uy
end

function z_vort!(spacevort, vort, u, plan, grid)
    apply!(z_vort_kernel!, grid, (vort, u, grid))
    ldiv!(spacevort, plan, vort)
    spacevort .*= grid.n^dim(grid) # FFT factor
    nothing
end

function spectral_stuff(grid; npoint = nothing)
    (; l, backend) = grid
    T = typeof(l)

    n = grid.n
    kmax = div(n, 2)

    kk = if dim(grid) == 2
        kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
        ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
        @. kx^2 + ky^2
    else
        kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
        ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
        kz = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, 1, :) # Normal FFT wavenumbers
        @. kx^2 + ky^2 + kz^2
    end
    kk = reshape(kk, :)

    isort = sortperm(kk) # Permutation for sorting the wavenumbers
    kksort = kk[isort]

    # Output query points (evenly log-spaced, but only integer wavenumbers)
    if isnothing(npoint)
        kuse = 1:kmax
    else
        kuse = logrange(T(1), T(kmax), npoint)
        kuse = sort(unique(round.(Int, kuse)))
    end

    # Since the wavenumbers are sorted, we just need to find the start and stop of each shell.
    # The linear indices for that shell is then given by the permutation in that range.
    inds = map(kuse) do k
        jstart = findfirst(≥(k^2), kksort)
        jstop = findfirst(≥((k + 1)^2), kksort)
        isnothing(jstop) && (jstop = length(kksort) + 1) # findfirst may return nothing
        jstop -= 1
        isort[jstart:jstop] # Linear indices of the i-th shell
    end

    # We need to adapt the shells for RFFT.
    # Consider the following example:
    #
    # julia> n, kmax = 8, 4;
    # julia> u = randn(n, n, n);
    # julia> f = fft(u); r = rfft(u);
    # julia> sum(abs2, f)
    # 275142.33506202063
    # julia> sum(abs2, r) + sum(abs2, view(r, 2:kmax, :, :))
    # 275142.3350620207
    #
    # To compute the energy of the FFT, we need an additional term for RFFT.
    # The second term sums over all the x-indices except for 1 and kmax + 1.
    # We thus need to add indices to account for the conjugate symmetry in RFFT.
    # For an RFFT array r of size (kmax + 1, n, n), we have the linear index relation
    # r[i] == r[x, y, z]
    # if
    # i == x + (y - 1) * (kmax + 1) + (z - 1) * (kmax + 1) * n.
    # We therefore need to exclude the indices:
    # (x == 1), i.e. (i % (kmax + 1) == 1), and
    # (x == kmax + 1), i.e. (i % (kmax + 1) == 0).
    # We only keep i if (i % (kmax + 1) > 1).
    conjinds = map(i -> filter(j -> j % (kmax + 1) > 1, i), inds)
    inds = map(vcat, inds, conjinds) # Include conjugate indices

    # Put indices on GPU
    inds = map(adapt(backend), inds)

    # Temporary arrays for spectrum computation
    e = KernelAbstractions.allocate(backend, T, ndrange(grid))

    (; shells = inds, k = kuse, e)
end

function spectrum(u, grid; npoint = nothing, stuff = spectral_stuff(grid; npoint))
    (; shells, k, e) = stuff
    if dim(grid) == 2
        @. e = abs2(u.x) / 2 + abs2(u.y) / 2
    else
        @. e = abs2(u.x) / 2 + abs2(u.y) / 2 + abs2(u.z) / 2
    end
    s = map(i -> sum(view(e, i)), shells) # Total energy in each shell
    (; k, s)
end

getenergy(u) = sum(abs2, u) + sum(abs2, selectdim(u, 1, 2:(size(u, 1)-1)))

@kernel function strainrate!(S, u, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumbers(g, I)
    ux, uy = u.x[I], u.y[I]
    S.xx[I] = im * kx * ux
    S.yy[I] = im * ky * uy
    S.xy[I] = im * (ky * ux + kx * uy) / 2
end
@kernel function strainrate!(S, u, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(g, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    S.xx[I] = im * kx * ux
    S.yy[I] = im * ky * uy
    S.zz[I] = im * kz * uz
    S.xy[I] = im * (ky * ux + kx * uy) / 2
    S.yz[I] = im * (kz * uy + ky * uz) / 2
    S.zx[I] = im * (kx * uz + kz * ux) / 2
end

function turbulence_statistics(u, visc, g)
    e = sum(getenergy, u) / 2
    uavg = sqrt(2 * e)
    S = tensorfield(g)
    apply!(strainrate!, g, (S, u, g))
    Sij = spacescalarfield(g)
    diss = if dim(g) == 2
        2 * visc * (getenergy(S.xx) + getenergy(S.yy) + 2 * getenergy(S.xy))
    else
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
    end
    l_kol = (visc^3 / diss)^(1 / 4)
    l_tay = sqrt(visc / diss) * uavg
    l_int = uavg^3 / diss
    t_int = l_int / uavg
    t_tay = l_tay / uavg
    t_kol = visc / diss |> sqrt
    Re_int = l_int * uavg / visc
    Re_tay = l_tay * uavg / visc
    Re_kol = l_kol * uavg / visc
    (; uavg, diss, l_int, l_tay, l_kol, t_int, t_tay, t_kol, Re_int, Re_tay, Re_kol)
end

getcache(grid) = (;
    ustart = vectorfield(grid),
    du = vectorfield(grid),
    σ = tensorfield(grid),
    vi_vj = spacescalarfield(grid),
    v = spacevectorfield(grid),
    plan = getplan(grid),
)

function forwardeuler!(u, cache, Δt, visc, grid)
    (; σ, vi_vj, v, plan, du) = cache
    stress!(σ, vi_vj, v, u, plan, visc, grid)
    apply!(tensordivergence!, grid, (du, σ, grid))
    for i = 1:dim(grid)
        axpy!(Δt, du[i], u[i])
    end
    apply!(project!, grid, (u, grid))
end

@kernel function abcn_kernel!(u, du, du_old, Δt, visc, g::Grid{2})
    I = @index(Global, Cartesian)
    kx, ky = wavenumbers(g, I)
    a = Δt / 2 * visc * (kx^2 + ky^2)
    u.x[I] = (1 - a) / (1 + a) * u.x[I] + Δt / (1 + a) * (3 * du.x[I] - du_old.x[I]) / 2
    u.y[I] = (1 - a) / (1 + a) * u.y[I] + Δt / (1 + a) * (3 * du.y[I] - du_old.y[I]) / 2
end
@kernel function abcn_kernel!(u, du, du_old, Δt, visc, g::Grid{3})
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(g, I)
    a = Δt / 2 * visc * (kx^2 + ky^2 + kz^2)
    u.x[I] = (1 - a) / (1 + a) * u.x[I] + Δt / (1 + a) * (3 * du.x[I] - du_old.x[I]) / 2
    u.y[I] = (1 - a) / (1 + a) * u.y[I] + Δt / (1 + a) * (3 * du.y[I] - du_old.y[I]) / 2
    u.z[I] = (1 - a) / (1 + a) * u.z[I] + Δt / (1 + a) * (3 * du.z[I] - du_old.z[I]) / 2
end

function abcn!(u, cache, Δt, visc, grid; firststep)
    (; σ, vi_vj, v, plan, du, ustart) = cache
    du_old = ustart # use same name as wray3 cache
    nonlinearity!(σ, vi_vj, v, u, plan, grid)
    apply!(tensordivergence!, grid, (du, σ, grid))
    firststep && foreach(copyto!, du_old, du) # Forward-Euler for first step
    apply!(abcn_kernel!, grid, (u, du, du_old, Δt, visc, grid))
    apply!(project!, grid, (u, grid))
    foreach(copyto!, du_old, du)
end

function propose_timestep(u, cache, visc, grid)
    (; vi_vj, du, v, plan) = cache
    D = dim(grid)
    for i = 1:D
        copyto!(du[i], u[i])
        ldiv!(v[i], plan, du[i]) # ldiv! overwrites input...
        v[i] .*= grid.n^D # FFT factor
    end
    D == 2 && @. vi_vj = sqrt(v.x^2 + v.y^2)
    D == 3 && @. vi_vj = sqrt(v.x^2 + v.y^2 + v.z^2)
    vmax = maximum(vi_vj)
    h = grid.l / grid.n
    Δt_conv = h / vmax
    Δt_diff = h^2 / D / 2 / visc
    min(Δt_conv, Δt_diff)
end

"Perform time step using Wray's third-order scheme."
function wray3!(u, cache, Δt, visc, grid)
    (; ustart, du, σ, vi_vj, v, plan) = cache
    T = eltype(u.x)
    D = dim(grid)

    # Low-storage Butcher tableau:
    # c1 | 0             ⋯   0
    # c2 | a1  0         ⋯   0
    # c3 | b1 a2  0      ⋯   0
    # c4 | b1 b2 a3  0   ⋯   0
    #  ⋮ | ⋮   ⋮  ⋮  ⋱   ⋱   ⋮
    # cn | b1 b2 b3  ⋯ an-1  0
    # ---+--------------------
    #    | b1 b2 b3  ⋯ bn-1 an
    #
    # Note the definition of (ai)i.
    # They are shifted to simplify the for-loop.
    # TODO: Make generic by passing a, b, c as inputs
    a = T(8 / 15), T(5 / 12), T(3 / 4)
    b = T(1 / 4), T(0)
    c = T(0), T(8 / 15), T(2 / 3)
    nstage = length(a)

    # Update current solution
    foreach(copyto!, ustart, u)

    for i = 1:nstage
        stress!(σ, vi_vj, v, u, plan, visc, grid)
        apply!(tensordivergence!, grid, (du, σ, grid))

        # Compute u = project(ustart + Δt * a[i] * du)
        i == 1 || foreach(copyto!, u, ustart) # Skip first iter
        for j = 1:D
            axpy!(a[i] * Δt, du[j], u[j])
        end
        apply!(project!, grid, (u, grid))

        # Compute ustart = ustart + Δt * b[i] * du
        # Skip last iter
        i < nstage && for j = 1:D
            axpy!(b[i] * Δt, du[j], ustart[j])
        end
    end

    u
end

function ouforcer(grid, kcut)
    kmax = div(grid.n, 2)
    D = dim(grid)
    kk = if D == 2
        kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
        ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
        @. kx^2 + ky^2
    else
        kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
        ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
        kz = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, 1, :) # Normal FFT wavenumbers
        @. kx^2 + ky^2 + kz^2
    end
    kk = reshape(kk, :)
    iuse = findall(kk -> 0 < kk < kcut^2, kk) # Exclude 0-th mode
    kuse = kk[iuse]
    nuse = length(iuse)
    x = complex(grid.l)
    b = KernelAbstractions.zeros(grid.backend, typeof(x), nuse, D)
    bold = zero(b)
    (; iuse, kuse, b, bold)
end

# @kernel function vectorgradient!(Gij, u, grid::Grid{3}, i, j)
#     I = @index(Global, Cartesian)
#     kk = wavenumbers(grid, I)
#     uu = u.x[I], u.y[I], u.z[I]
#     Gij[I] = im * kk[j] * uu[i]
# end

# @kernel function qcrit!(q, G, grid::Grid{3})
#     I = @index(Global, Cartesian)
#     g = G.xx[I]
# end

export Grid, apply!, dim, tensordim
export scalarfield, vectorfield, tensorfield, randomfield
export getcache, propose_timestep, project!, stress!, tensordivergence!

end
