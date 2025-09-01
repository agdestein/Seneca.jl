"Pseudo-spectral solver for the 3D incompressible Navier-Stokes equations."
module Seneca

using AbstractFFTs
using Adapt
using FFTW
using LinearAlgebra
using KernelAbstractions
using Random

@kwdef struct Grid{T,B}
    "Domain side length."
    l::T

    "Number of grid points in each dimension."
    n::Int

    """
    KernelAbstractions.jl hardware backend.
    For Nvidia GPUs, do `using CUDA` and set to `CUDABackend()`.
    """
    backend::B = CPU()

    "Kernel work group size."
    workgroupsize::Int = 64
end

# Like fftfreq, but with proper type
@inline fftfreq_int(g::Grid, i::Int) = i - 1 - ifelse(i <= (g.n + 1) >> 1, 0, g.n)
@inline fftfreq_int(g::Grid, I::CartesianIndex) =
    I.I[1] - 1, fftfreq_int(g, I.I[2]), fftfreq_int(g, I.I[3])
@inline wavenumbers(g::Grid, I::CartesianIndex) = (
    pi / g.l * 2 * (I.I[1] - 1),
    pi / g.l * 2 * fftfreq_int(g, I.I[2]),
    pi / g.l * 2 * fftfreq_int(g, I.I[3]),
)
ndrange(n) = div(n, 2) + 1, n, n

function apply!(kernel!, grid, args...; ndrange = ndrange(grid.n))
    (; backend, workgroupsize) = grid
    kernel!(backend, workgroupsize)(args...; ndrange)
    KernelAbstractions.synchronize(backend)
    nothing
end

scalarfield(g::Grid{T}) where {T} =
    KernelAbstractions.zeros(g.backend, Complex{T}, ndrange(g.n))
vectorfield(grid) = (; x = scalarfield(grid), y = scalarfield(grid), z = scalarfield(grid))
tensorfield(grid) = (;
    xx = scalarfield(grid),
    yy = scalarfield(grid),
    zz = scalarfield(grid),
    xy = scalarfield(grid),
    yz = scalarfield(grid),
    zx = scalarfield(grid),
)

spacescalarfield(g::Grid{T}) where {T} =
    KernelAbstractions.zeros(g.backend, T, (g.n, g.n, g.n))
spacevectorfield(grid) =
    (; x = spacescalarfield(grid), y = spacescalarfield(grid), z = spacescalarfield(grid))

@kernel function project!(u, grid)
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(grid, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    p = (kx * ux + ky * uy + kz * uz) / (kx * kx + ky * ky + kz * kz)
    p = ifelse(I.I == (1, 1, 1), zero(p), p) # Leave constant mode intact
    u.x[I] = ux - kx * p
    u.y[I] = uy - ky * p
    u.z[I] = uz - kz * p
end

@kernel function twothirds!(ubar, u, grid)
    I = @index(Global, Cartesian)
    kx, ky, kz = fftfreq_int(grid, I)
    K = div(grid.n, 2)
    ix = 3 * abs(kx) ≤ 2 * K
    iy = 3 * abs(ky) ≤ 2 * K
    iz = 3 * abs(kz) ≤ 2 * K
    ubar[I] = ifelse(ix & iy & iz, u[I], zero(eltype(u)))
    # ubar[I] = ifelse(9 * (kx^2 + ky^2 + kz^2) ≤ 4 * K^2, u[I], zero(eltype(u)))
end

getplan(grid) = plan_rfft(spacescalarfield(grid))

function nonlinearity!(σ, vi_vj, v, u, plan, grid)
    temp = σ.xx # Use σ.xx as temporary complex storage
    for i = 1:3
        apply!(twothirds!, grid, temp, u[i], grid) # Zero out high wavenumbers
        ldiv!(v[i], plan, temp) # Inverse transform
        v[i] .*= grid.n^3
    end
    #! format: off
    @. vi_vj = v.x * v.x; mul!(σ.xx, plan, vi_vj); (σ.xx ./= grid.n^3)
    @. vi_vj = v.y * v.y; mul!(σ.yy, plan, vi_vj); (σ.yy ./= grid.n^3)
    @. vi_vj = v.z * v.z; mul!(σ.zz, plan, vi_vj); (σ.zz ./= grid.n^3)
    @. vi_vj = v.x * v.y; mul!(σ.xy, plan, vi_vj); (σ.xy ./= grid.n^3)
    @. vi_vj = v.y * v.z; mul!(σ.yz, plan, vi_vj); (σ.yz ./= grid.n^3)
    @. vi_vj = v.z * v.x; mul!(σ.zx, plan, vi_vj); (σ.zx ./= grid.n^3)
    #! format: on
    nothing
end

@kernel function viscosity!(σ, u, visc, grid)
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(grid, I)
    ux, uy, uz = u.x[I], u.y[I], u.z[I]
    σ.xx[I] -= visc * im * kx * ux
    σ.yy[I] -= visc * im * ky * uy
    σ.zz[I] -= visc * im * kz * uz
    σ.xy[I] -= visc * im * (ky * ux + kx * uy)
    σ.yz[I] -= visc * im * (kz * uy + ky * uz)
    σ.zx[I] -= visc * im * (kx * uz + kz * ux)
end

function stress!(σ, vi_vj, v, u, plan, visc, grid)
    # foreach(s -> fill!(s, 0), σ)
    nonlinearity!(σ, vi_vj, v, u, plan, grid)
    apply!(viscosity!, grid, σ, u, visc, grid)
end

@kernel function vectordivergence!(div, u, grid)
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(grid, I)
    div[I] = im * kx * u.x[I] + im * ky * u.y[I] + im * kz * u.z[I]
end

@kernel function tensordivergence!(div, σ, grid)
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(grid, I)
    div.x[I] = -im * kx * σ.xx[I] - im * ky * σ.xy[I] - im * kz * σ.zx[I]
    div.y[I] = -im * kx * σ.xy[I] - im * ky * σ.yy[I] - im * kz * σ.yz[I]
    div.z[I] = -im * kx * σ.zx[I] - im * ky * σ.yz[I] - im * kz * σ.zz[I]
end

profile(k, kpeak) = k^4 * exp(-2 * (k / kpeak)^2)

"Taylor-Green vortex."
function taylorgreen(grid, plan)
    (; l, n, backend) = grid
    x = range(zero(l), one(l), n + 1)[2:end] |> Array |> adapt(backend)
    y = reshape(x, 1, :)
    z = reshape(x, 1, 1, :)
    v = spacescalarfield(grid)
    #! format: off
    @. v =  sinpi(2x) * cospi(2y) * sinpi(2z) / 2; ux = plan * v; ux ./= n^3
    @. v = -cospi(2x) * sinpi(2y) * sinpi(2z) / 2; uy = plan * v; uy ./= n^3
    #! format: on
    v = nothing
    uz = zero(ux)
    u = (; x = ux, y = uy, z = uz)
    apply!(project!, grid, u, grid)
    u
end

"""
Make random velocity field with prescribed energy spectrum profile.

Note: The profile takes the scalar wavenumber norm as input,
define it as `profile(k)`.
"""
function randomfield(grid; kpeak = 5, totalenergy = 1, rng = Random.default_rng())
    # Compute energy
    @kernel function energy!(E, u, n)
        I = @index(Global, Cartesian)
        E[I] = (abs2(u.x[I]) + abs2(u.y[I]) + abs2(u.z[I])) / 2
    end

    # Mask for active wavenumbers: kleft ≤ k < kleft + 1
    # Do everything squared to avoid floats
    @kernel function mask!(mask, kleft, n)
        I = @index(Global, Cartesian)
        kx, ky, kz = fftfreq_int(grid, I)
        mask[I] = kleft^2 ≤ kx^2 + ky^2 + kz^2 < (kleft + 1)^2
    end

    # Adjust the amplitude to match energy profile
    @kernel function normalize!(u, mask, factor)
        I = @index(Global, Cartesian)
        m = mask[I]
        ux, uy, uz = u.x[I], u.y[I], u.z[I]
        u.x[I] = ifelse(m, factor * ux, ux)
        u.y[I] = ifelse(m, factor * uy, uy)
        u.z[I] = ifelse(m, factor * uz, uz)
    end

    # Create random field and make it divergence free
    u = vectorfield(grid)
    foreach(u -> randn!(rng, u), u)
    apply!(project!, grid, u, grid)

    # RFFT exploits conjugate symmetry, so we only need half the modes
    kmax = div(grid.n, 2)
    range = ndrange(grid.n)

    # Allocate arrays
    E = similar(u.x, range...)
    Emask = similar(E)
    mask = similar(E, Bool)

    # Compute energy
    apply!(energy!, grid, E, u, grid.n; ndrange = range)

    # Maximum partially resolved wavenumber (sqrt(dim) * kmax)
    kdiag = floor(Int, sqrt(3) * div(grid.n, 2))

    # Sum of shell weights 
    totalprofile = sum(k -> profile(k, kpeak), 0:kdiag)

    # Adjust energy in each partially resolved shell [k, k+1)
    for k = 0:kdiag
        apply!(mask!, grid, mask, k, grid.n; ndrange = range) # Shell mask
        @. Emask = mask * E
        Eshell = sum(Emask) + sum(view(Emask, 2:kmax, :, :)) # Current energy in shell
        E0 = totalenergy * profile(k, kpeak) / totalprofile # Desired energy in shell
        factor = sqrt(E0 / Eshell) # E = u^2 / 2
        apply!(normalize!, grid, u, mask, factor; ndrange = range)
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
    E = @. (abs2(u.x) + abs2(u.y) + abs2(u.z)) / 2
    sum(E) + sum(view(E, 2:kmax, :, :))
end

@kernel function z_vort_kernel!(vort, u, grid)
    I = @index(Global, Cartesian)
    kx, ky, _ = wavenumbers(grid, I)
    ux, uy = u.x[I], u.y[I]
    vort[I] = -im * ky * ux + im * kx * uy
end

function z_vort!(spacevort, vort, u, plan, grid)
    apply!(z_vort_kernel!, grid, vort, u, grid)
    ldiv!(spacevort, plan, vort)
    spacevort .*= grid.n^3 # FFT factor
    nothing
end

function spectral_stuff(grid; npoint = nothing)
    (; l, backend) = grid
    T = typeof(l)

    n = grid.n
    kmax = div(n, 2)

    kx = 0:kmax # For RFFT, the x-wavenumbers are 0:kmax
    ky = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, :) # Normal FFT wavenumbers
    kz = reshape(map(i -> fftfreq_int(grid, i), 1:grid.n), 1, 1, :) # Normal FFT wavenumbers
    kk = @. kx^2 + ky^2 + kz^2
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
    e = KernelAbstractions.allocate(backend, T, kmax + 1, n, n)

    (; shells = inds, k = kuse, e)
end

function spectrum(u, grid; npoint = nothing, stuff = spectral_stuff(grid; npoint))
    (; shells, k, e) = stuff
    @. e = abs2(u.x) / 2 + abs2(u.y) / 2 + abs2(u.z) / 2
    s = map(i -> sum(view(e, i)), shells) # Total energy in each shell
    (; k, s)
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
    apply!(tensordivergence!, grid, du, σ, grid)
    axpy!(Δt, du.x, u.x)
    axpy!(Δt, du.y, u.y)
    axpy!(Δt, du.z, u.z)
    apply!(project!, grid, u, grid)
end

@kernel function abcn_kernel!(u, du, du_old, Δt, visc, grid)
    I = @index(Global, Cartesian)
    kx, ky, kz = wavenumbers(grid, I)
    a = Δt / 2 * visc * (kx^2 + ky^2 + kz^2)
    u.x[I] = (1 - a) / (1 + a) * u.x[I] + Δt / (1 + a) * (3 * du.x[I] - du_old.x[I]) / 2
    u.y[I] = (1 - a) / (1 + a) * u.y[I] + Δt / (1 + a) * (3 * du.y[I] - du_old.y[I]) / 2
    u.z[I] = (1 - a) / (1 + a) * u.z[I] + Δt / (1 + a) * (3 * du.z[I] - du_old.z[I]) / 2
end

function abcn!(u, cache, Δt, visc, grid; firststep)
    (; σ, vi_vj, v, plan, du, ustart) = cache
    du_old = ustart # use same name as wray3 cache
    nonlinearity!(σ, vi_vj, v, u, plan, grid)
    apply!(tensordivergence!, grid, du, σ, grid)
    firststep && foreach(copyto!, du_old, du) # Forward-Euler for first step
    apply!(abcn_kernel!, grid, u, du, du_old, Δt, visc, grid)
    apply!(project!, grid, u, grid)
    foreach(copyto!, du_old, du)
end

function propose_timestep(u, cache, visc, grid)
    (; vi_vj, du, v, plan) = cache
    for i = 1:3
        copyto!(du[i], u[i])
        ldiv!(v[i], plan, du[i])
    end
    @. vi_vj = sqrt(v.x^2 + v.y^2 + v.z^2)
    vmax = maximum(vi_vj)
    h = grid.l / grid.n
    Δt_conv = h / vmax
    Δt_diff = h^2 / 3 / 2 / visc
    min(Δt_conv, Δt_diff)
end

"Perform time step using Wray's third-order scheme."
function wray3!(u, cache, Δt, visc, grid)
    (; ustart, du, σ, vi_vj, v, plan) = cache
    T = eltype(u.x)

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
        apply!(tensordivergence!, grid, du, σ, grid)

        # Compute u = project(ustart + Δt * a[i] * du)
        i == 1 || foreach(copyto!, u, ustart) # Skip first iter
        axpy!(a[i] * Δt, du.x, u.x)
        axpy!(a[i] * Δt, du.y, u.y)
        axpy!(a[i] * Δt, du.z, u.z)
        apply!(project!, grid, u, grid)

        # Compute ustart = ustart + Δt * b[i] * du
        # Skip last iter
        if i < nstage
            axpy!(b[i] * Δt, du.x, ustart.x)
            axpy!(b[i] * Δt, du.y, ustart.y)
            axpy!(b[i] * Δt, du.z, ustart.z)
        end
    end

    u
end

end
