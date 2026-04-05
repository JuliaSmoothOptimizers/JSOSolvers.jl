export R2NMAB, R2NMABSolver, R2NArm

"""
    R2NArm{T}

A specific hyperparameter configuration (an "arm") for the MAB-R2N solver.
"""
struct R2NArm{T}
    θ1::T
    θ2::T
    η1::T
    η2::T
    γ1::T
    γ2::T
    γ3::T
end

"""
    R2NMABSolver

A mutable solver structure for the Bandit-tuned R2N algorithm. 
Maintains the standard R2N memory allocations while adding the UCB Bandit 
trackers (N counts, Q values) and composite reward weights.
"""
mutable struct R2NMABSolver{T, V, Sub <: AbstractR2NSubsolver{T}, M <: AbstractNLPModel{T, V}} <: AbstractOptimizationSolver
    x::V
    xt::V
    gx::V
    rhs::V
    y::V
    Hs::V
    s::V
    scp::V
    obj_vec::V
    subsolver::Sub
    h::LineModel{T, V, M}
    subtol::T
    σ::T
    params::R2NParameterSet{T}
    
    # Bandit Specific State
    arms::Vector{R2NArm{T}}
    N::Vector{Int}      # Play counts
    Q::Vector{T}        # Estimated average rewards
    ucb_c::T            # Exploration constant
    w1::T               # Weight: Model Agreement (ρ)
    w2::T               # Weight: Stationarity Progress
    w3::T               # Weight: Step Vigor
end

function R2NMABSolver(
    nlp::AbstractNLPModel{T, V},
    arms::Vector{R2NArm{T}};
    ucb_c::T = T(0.1),
    w1::T = T(0.4),
    w2::T = T(0.4),
    w3::T = T(0.2),
    δ1 = get(R2N_δ1, nlp),
    σmin = get(R2N_σmin, nlp),
    non_mono_size = get(R2N_non_mono_size, nlp),
    subsolver::Union{Type, AbstractR2NSubsolver} = CGR2NSubsolver(nlp),
    ls_c = get(R2N_ls_c, nlp),
    ls_increase = get(R2N_ls_increase, nlp),
    ls_decrease = get(R2N_ls_decrease, nlp),
    ls_min_alpha = get(R2N_ls_min_alpha, nlp),
    ls_max_alpha = get(R2N_ls_max_alpha, nlp),
    kwargs...
) where {T, V}
    
    @assert abs(w1 + w2 + w3 - one(T)) < sqrt(eps(T)) "Bandit weights w1, w2, w3 must sum to 1.0"
    @assert length(arms) > 0 "Must provide at least one R2NArm"

    # Default parameters for baseline fallback
    params = R2NParameterSet(
        nlp;
        δ1 = δ1,
        σmin = σmin,
        non_mono_size = non_mono_size,
        ls_c = ls_c,
        ls_increase = ls_increase,
        ls_decrease = ls_decrease,
        ls_min_alpha = ls_min_alpha,
        ls_max_alpha = ls_max_alpha,
    )

    nvar = nlp.meta.nvar
    x = V(undef, nvar); x .= nlp.meta.x0
    xt = V(undef, nvar)
    gx = V(undef, nvar)
    rhs = V(undef, nvar)
    y = isa(nlp, QuasiNewtonModel) ? V(undef, nvar) : V(undef, 0)
    Hs = V(undef, nvar)
    s = V(undef, nvar)
    scp = V(undef, nvar)
    
    σ = zero(T)
    subtol = one(T)
    obj_vec = fill(typemin(T), value(params.non_mono_size))
    h = LineModel(nlp, x, s)

    sub_instance = subsolver isa Type ? subsolver(nlp) : subsolver

    num_arms = length(arms)
    N = zeros(Int, num_arms)
    Q = zeros(T, num_arms)

    return R2NMABSolver{T, V, typeof(sub_instance), typeof(nlp)}(
        x, xt, gx, rhs, y, Hs, s, scp, obj_vec, sub_instance, h, subtol, σ, params,
        arms, N, Q, ucb_c, w1, w2, w3
    )
end

function SolverCore.reset!(solver::R2NMABSolver{T}) where {T}
    fill!(solver.obj_vec, typemin(T))
    fill!(solver.N, 0)
    fill!(solver.Q, zero(T))
    if solver.subsolver isa KrylovR2NSubsolver
        LinearOperators.reset!(solver.subsolver.H)
    end
    return solver
end

function SolverCore.reset!(solver::R2NMABSolver{T}, nlp::AbstractNLPModel) where {T}
    fill!(solver.obj_vec, typemin(T))
    fill!(solver.N, 0)
    fill!(solver.Q, zero(T))
    if solver.subsolver isa KrylovR2NSubsolver
        LinearOperators.reset!(solver.subsolver.H)
    end
    solver.h = LineModel(nlp, solver.x, solver.s)
    return solver
end

"""
    R2NMAB(nlp, arms; kwargs...)

Executes the R2N algorithm with online hyperparameter tuning via Upper Confidence Bound (UCB) Multi-Armed Bandits.
"""
function R2NMAB(
    nlp::AbstractNLPModel{T, V},
    arms::Vector{R2NArm{T}};
    kwargs...
) where {T, V}
    solver = R2NMABSolver(nlp, arms; kwargs...)
    return solve!(solver, nlp; kwargs...)
end

function SolverCore.solve!(
    solver::R2NMABSolver{T, V},
    nlp::AbstractNLPModel{T, V},
    stats::GenericExecutionStats{T, V} = GenericExecutionStats(nlp);
    callback = (args...) -> nothing,
    x::V = nlp.meta.x0,
    atol::T = √eps(T),
    rtol::T = √eps(T),
    max_time::Float64 = 30.0,
    max_eval::Int = -1,
    max_iter::Int = typemax(Int),
    verbose::Int = 0,
    subsolver_verbose::Int = 0,
    npc_handler::Symbol = :ag,
    scp_flag::Bool = true,
    always_accept_npc_ag::Bool = false,
    fast_local_convergence::Bool = false,
    kwargs...
) where {T, V}
    
    unconstrained(nlp) || error("R2NMAB should only be called on unconstrained problems.")
    
    SolverCore.reset!(stats)
    params = solver.params
    δ1 = value(params.δ1)
    σmin = value(params.σmin)
    non_mono_size = value(params.non_mono_size)

    ls_c = value(params.ls_c)
    ls_increase = value(params.ls_increase)
    ls_decrease = value(params.ls_decrease)

    start_time = time()
    set_time!(stats, 0.0)

    n = nlp.meta.nvar
    x = solver.x .= x
    xt = solver.xt
    ∇fk = solver.gx
    rhs = solver.rhs
    y = solver.y
    s = solver.s
    scp = solver.scp
    Hs = solver.Hs
    σk = solver.σ

    subtol = solver.subtol

    initialize!(solver.subsolver, nlp, x)
    H = get_operator(solver.subsolver)

    set_iter!(stats, 0)
    f0 = obj(nlp, x)
    set_objective!(stats, f0)

    grad!(nlp, x, ∇fk)
    norm_∇fk = norm(∇fk)
    set_dual_residual!(stats, norm_∇fk)

    σk = 2^round(log2(norm_∇fk + 1)) / norm_∇fk
    ρk = zero(T)

    fmin = min(-one(T), f0) / eps(T)
    unbounded = f0 < fmin

    ϵ = atol + rtol * norm_∇fk
    optimal = norm_∇fk ≤ ϵ

    if optimal
        set_status!(stats, :first_order)
        set_solution!(stats, x)
        return stats
    end

    subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
    solver.σ = σk
    solver.subtol = subtol

    callback(nlp, solver, stats)
    subtol = solver.subtol
    σk = solver.σ

    done = stats.status != :unknown
    ft = f0

    step_accepted = false
    sub_stats = :unknown
    subiter = 0
    dir_stat = ""
    is_npc_ag_step = false

    while !done
        # ==========================================================
        # 1. BANDIT ACTION SELECTION (UCB)
        # ==========================================================
        unplayed_idx = findfirst(==(0), solver.N)
        i_k = if unplayed_idx !== nothing
            unplayed_idx # Pure exploration of all arms first
        else
            total_plays = sum(solver.N)
            # argmax (Q_i + c * sqrt(ln(t) / N_i))
            argmax(solver.Q .+ solver.ucb_c .* sqrt.(log(T(total_plays)) ./ solver.N))
        end

        # Extract dynamic parameters for this iteration
        active_arm = solver.arms[i_k]
        θ1 = active_arm.θ1
        θ2 = active_arm.θ2
        η1 = active_arm.η1
        η2 = active_arm.η2
        γ1 = active_arm.γ1
        γ2 = active_arm.γ2
        γ3 = active_arm.γ3

        # Track old state for Bandit Reward
        norm_∇fk_old = norm_∇fk
        
        # ==========================================================
        # 2. CORE R2N STEP (Using dynamic parameters)
        # ==========================================================
        npcCount = 0
        fck_computed = false

        @. rhs = -∇fk

        subsolver_solved, sub_stats, subiter, npcCount = 
            solver.subsolver(s, rhs, σk, atol, subtol, n; verbose = subsolver_verbose)

        if !subsolver_solved && npcCount == 0
            set_status!(stats, :stalled)
            done = true
            break
        end

        calc_scp_needed = false
        force_sigma_increase = false
        if solver.subsolver isa HSLR2NSubsolver
            num_neg, num_zero = get_inertia(solver.subsolver)
            if num_zero > 0
                force_sigma_increase = true
            end
            if !force_sigma_increase && num_neg > 0
                mul!(Hs, H, s)
                if dot(s, Hs) < 0
                    npcCount = 1
                    if npc_handler == :prev
                        npc_handler = :ag
                    end
                else
                    calc_scp_needed = true
                end
            end
        end

        if !(solver.subsolver isa ShiftedLBFGSSolver) && npcCount >= 1
            if npc_handler == :ag
                is_npc_ag_step = true
                npcCount = 0
                dir = solver.subsolver isa HSLR2NSubsolver ? s : get_npc_direction(solver.subsolver)

                SolverTools.redirect!(solver.h, x, dir)
                α, ft, _, _ = armijo_goldstein(
                    solver.h, stats.objective, dot(∇fk, dir);
                    t = one(T), τ₀ = ls_c, τ₁ = 1 - ls_c, γ₀ = ls_decrease, γ₁ = ls_increase,
                    bk_max = 100, bG_max = 100, verbose = (verbose > 0)
                )
                @. s = α * dir
                fck_computed = true
            elseif npc_handler == :prev
                npcCount = 0
            end
        end

        if scp_flag == true || npc_handler == :cp || calc_scp_needed
            mul!(Hs, H, ∇fk)
            γ_k_c = (dot(∇fk, Hs) + σk * norm_∇fk^2) / norm_∇fk^2

            if γ_k_c > 0
                ν_k = 2 * (1 - δ1) / γ_k_c
            else
                λmax = get_operator_norm(solver.subsolver)
                ν_k = θ1 / (λmax + σk)
            end

            @. scp = -ν_k * ∇fk

            if (npc_handler == :cp && npcCount >= 1) || (norm(s) > θ2 * norm(scp))
                npcCount = 0
                s .= scp
                fck_computed = false
            end
        end

        if force_sigma_increase || (npc_handler == :sigma && npcCount >= 1)
            step_accepted = false
            σk = γ2 * σk
            npcCount = 0
        else 
            if is_npc_ag_step && always_accept_npc_ag
                is_npc_ag_step = false
                ρk = η1 
                @. xt = x + s
                if !fck_computed
                    ft = obj(nlp, xt)
                end
                step_accepted = true
            else
                mul!(Hs, H, s)
                ΔTk = -dot(s, ∇fk) - dot(s, Hs) / 2

                if ΔTk <= eps(T) * max(one(T), abs(stats.objective))
                    step_accepted = false
                else
                    @. xt = x + s
                    if !fck_computed
                        ft = obj(nlp, xt)
                    end

                    if non_mono_size > 1
                        k_idx = mod(stats.iter, non_mono_size) + 1
                        solver.obj_vec[k_idx] = stats.objective
                        ft_max = maximum(solver.obj_vec)
                        ρk = (ft_max - ft) / (ft_max - stats.objective + ΔTk)
                    else
                        ρk = (stats.objective - ft) / ΔTk
                    end
                    step_accepted = ρk >= η1
                end
            end  

            if step_accepted
                if isa(nlp, QuasiNewtonModel)
                    rhs .= ∇fk 
                end

                x .= xt
                grad!(nlp, x, ∇fk) 

                if isa(nlp, QuasiNewtonModel)
                    @. y = ∇fk - rhs 
                    push!(nlp, s, y)
                end

                if !(solver.subsolver isa ShiftedLBFGSSolver)
                    update_subsolver!(solver.subsolver, nlp, x)
                    H = get_operator(solver.subsolver)
                end

                set_objective!(stats, ft)
                unbounded = ft < fmin
                norm_∇fk = norm(∇fk)

                if ρk >= η2
                    σk = fast_local_convergence ? γ3 * min(σk, norm_∇fk) : γ3 * σk
                else
                    σk = γ1 * σk
                end
            else
                σk = γ2 * σk
            end
        end

        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)

        subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
        set_dual_residual!(stats, norm_∇fk)

        solver.σ = σk
        solver.subtol = subtol

        callback(nlp, solver, stats)
        
        # ==========================================================
        # 3. BANDIT REWARD & BELIEF UPDATE
        # ==========================================================
        norm_s = norm(s)
        
        # Component 1: Bounded Model Agreement (Clip to [0,1] to avoid wild swings)
        rho_reward = max(zero(T), min(one(T), ρk))
        
        # Component 2: Stationarity Progress
        # If rejected, norm_∇fk is unchanged, so progress is 0.
        grad_progress = max(zero(T), (norm_∇fk_old - norm_∇fk) / norm_∇fk_old)
        
        # Component 3: Step Vigor 
        step_vigor = norm_s / (one(T) + norm_s)
        
        # Composite calculation
        raw_reward = solver.w1 * rho_reward + solver.w2 * grad_progress + solver.w3 * step_vigor
        final_reward = max(zero(T), raw_reward)
        
        # Update trackers for the chosen arm
        solver.N[i_k] += 1
        solver.Q[i_k] += (final_reward - solver.Q[i_k]) / solver.N[i_k]

        # ==========================================================
        # Loop Termination Checking
        # ==========================================================
        norm_∇fk = stats.dual_feas
        σk = solver.σ
        optimal = norm_∇fk ≤ ϵ

        if stats.status == :user
            done = true
        else
            set_status!(
                stats,
                get_status(
                    nlp,
                    elapsed_time = stats.elapsed_time,
                    optimal = optimal,
                    unbounded = unbounded,
                    max_eval = max_eval,
                    iter = stats.iter,
                    max_iter = max_iter,
                    max_time = max_time,
                ),
            )
            done = stats.status != :unknown
        end
    end

    set_solution!(stats, x)
    return stats
end