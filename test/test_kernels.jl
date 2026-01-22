# set rotational and macroturbulence 
vsini = 2100.0
ζ_rt = 1400.0

# set limb darkening
u1 = 0.4
u2 = 0.0

xs = range(6301.0, 6310.0, step=0.005)
# ys = Array(cfunc_flux_stationary.cfunc_dt)
intres = 1024

N = length(xs)
λ0 = mean(xs)
vs = FT.c_ms .* (xs .- λ0) ./ λ0
Δv = (last(vs) - first(vs)) / (N - 1)
dv = diff(vs)

shift = -1

# hirano kernel no rot
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, 0.0, ζ_rt; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_no_rot = circshift(k_ctr ./ sum(k_ctr), shift)

# hirano kernel no mac
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, 0.0; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_no_macro = circshift(k_ctr ./ sum(k_ctr), shift)

# hirano rotmacro
σ = FFTW.fftfreq(N) ./ Δv
Kσ = FT.hirano_rotmacro_ft_kernel(σ, vsini, ζ_rt; u1=u1, u2=u2, intres=intres)
K_dft = Kσ ./ Δv
k_circ = real(ifft(K_dft))
k_ctr  = FFTW.fftshift(k_circ)
n = collect(-div(N,2):(N-1-div(N,2))) 
v_ctr = n .* Δv
hirano_rot_macro = circshift(k_ctr ./ sum(k_ctr), shift)

# get the gray rt kernel and rotation kernel
iso_rt_macro_kernel = FT.gray_iso_rt_macro_kernel(vs, ζ_rt)
gray_rot_kernel = FT.gray_rot_kernel(vs, vsini, u1)

# do the testing
@testset "Testing kernel outputs" begin
    @test maximum(abs.(hirano_no_rot .- iso_rt_macro_kernel)) < 0.5
    @test maximum(abs.(gray_rot_kernel .- hirano_no_macro)) < 0.5
end

# # get isotropic gaussian
# σ_g(x) = x * (ζ_rt / FT.c_ms)
# g(x, n) = exp(-((x - n) / σ_g(x))^2.0)

# # offset the kernel by the velocity
# λ0 = mean(λs_korg)
# λc = λ0

# # sample the kernel
# gaussian = g.(λs_korg, λc)
# gaussian ./= sum(gaussian)

# # plot the RT case
# plt.close("all")
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
# ax1.plot(λs_korg, iso_rt_macro_kernel, label="gray")
# ax1.plot(λs_korg, hirano_no_rot, label="hirano")
# ax2.scatter(λs_korg, hirano_no_rot .- iso_rt_macro_kernel, c="tab:blue", s=2)
# # ax1.set_xlim(6301.8, 6302.2)
# ax1.legend()
# ax1.set_title("Macro Only")
# plt.savefig("derp.pdf")
# plt.show()

# # plot the vsini case
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
# ax1.plot(λs_korg, gray_rot_kernel, label="gray")
# ax1.plot(λs_korg, hirano_no_macro, label="hirano")
# ax2.scatter(λs_korg, hirano_no_macro .- gray_rot_kernel, c="tab:blue", s=2)
# ax1.set_xlim(6301.8, 6302.2)
# ax1.set_title("Rotation Only")
# ax1.legend()
# plt.show()

# # now get contribution functions + flux
# cfunc_flux_hirano_norot = FT.convolve_hirano_rotmacro(xs, ys, 0.0, ζ_rt, u1, u2, intres=intres)
# cfunc_flux_hirano_nomacro = FT.convolve_hirano_rotmacro(xs, ys, vsini, 0.0, u1, u2, intres=intres)
# cfunc_flux_hirano_rotmacro = FT.convolve_hirano_rotmacro(xs, ys, vsini, ζ_rt, u1, u2, intres=intres)

# cfunc_flux_rotgray = FT.convolve_gray_rotation(xs, ys, vsini, u1)
# cfunc_flux_macrogray = FT.convolve_iso_rt_macro(xs, ys, ζ_rt)

# flux_hirano_norot = dropdims(sum(cfunc_flux_hirano_norot, dims=1), dims=1)
# flux_hirano_nomacro = dropdims(sum(cfunc_flux_hirano_nomacro, dims=1), dims=1)

# flux_rotgray = dropdims(sum(cfunc_flux_rotgray, dims=1), dims=1)
# flux_macrogray = dropdims(sum(cfunc_flux_macrogray, dims=1), dims=1)

# # plot the RT case
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
# ax1.plot(λs_korg, flux_macrogray, label="gray")
# ax1.plot(λs_korg, flux_hirano_norot, label="hirano")
# ax2.scatter(λs_korg, 100 .* (flux_hirano_norot .- flux_macrogray) ./ flux_hirano_norot, c="tab:blue", s=2)
# ax1.legend()
# ax1.set_title("Macro Only")
# plt.show()

# # plot the vsini case
# fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1, sharex=true, height_ratios=[4,1])
# ax1.plot(λs_korg, flux_rotgray, label="gray")
# ax1.plot(λs_korg, flux_hirano_nomacro, label="hirano")
# ax2.scatter(λs_korg, 100 .* (flux_hirano_nomacro .- flux_rotgray) ./ flux_hirano_nomacro, c="tab:blue", s=2)
# ax1.legend()
# ax1.set_title("Rotation Only")
# plt.show()
