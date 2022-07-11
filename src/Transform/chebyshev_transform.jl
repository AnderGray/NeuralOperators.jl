export ChebyshevTransform

struct ChebyshevTransform{N, S}<:AbstractTransform
    modes::NTuple{N, S} # N == ndims(x)
end

Base.ndims(::ChebyshevTransform{N}) where {N} = N

function transform(t::ChebyshevTransform{N}, 𝐱::AbstractArray) where {N}
    return FFTW.r2r(𝐱, FFTW.REDFT10, 1:N) # [size(x)..., in_chs, batch]
end

function truncate_modes(t::ChebyshevTransform, 𝐱̂::AbstractArray)
    return view(𝐱̂, map(d->1:d, t.modes)..., :, :) # [t.modes..., in_chs, batch]
end

function inverse(t::ChebyshevTransform{N}, 𝐱̂::AbstractArray) where {N}
    normalized_𝐱̂ = 𝐱̂ ./ (prod(2 .* (size(𝐱̂)[1:N] .- 1)))
    return FFTW.r2r(normalized_𝐱̂, FFTW.REDFT01, 1:N) # [size(x)..., in_chs, batch]
end

function ChainRulesCore.rrule(::typeof(FFTW.r2r), x::AbstractArray, kind, dims)
    y = FFTW.r2r(x, kind, dims)
    (M,) = size(x)[dims]
    r2r_pullback(Δ) = (NoTangent(), ∇r2r(unthunk(Δ), kind, dims, M), NoTangent(), NoTangent())
    return y, r2r_pullback
end

function ∇r2r(Δ::AbstractArray, kind, dims, M)
    # derivative of r2r turns out to be r2r + a rank 4 correction
    Δx = FFTW.r2r(Δ, kind, dims)
    
    # a1 = fill!(similar(A, M), one(T))
    # CUDA.@allowscalar a1[1] = a1[end] = zero(T)

    # a2 = fill!(similar(A, M), one(T))
    # a2[1:2:end] .= -one(T)
    # CUDA.@allowscalar a2[1] = a2[end] = zero(T)

    # e1 = fill!(similar(A, M), zero(T))
    # CUDA.@allowscalar e1[1] = one(T)

    # eN = fill!(similar(A, M), zero(T))
    # CUDA.@allowscalar eN[end] = one(T)

    # @tullio Δx[s, i, b] +=
    #     a1[i] * e1[k] * Δ[s, k, b] - a2[i] * eN[k] * Δ[s, k, b]
    # @tullio Δx[s, i, b] +=
    #     eN[i] * a2[k] * Δ[s, k, b] - e1[i] * a1[k] * Δ[s, k, b]
    return Δx
end
