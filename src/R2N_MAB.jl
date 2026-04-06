export R2NMAB, R2NContextualSolver, R2NArm

"""
    R2NArm{T}

A specific regularization configuration (an "arm") for the Contextual MAB-R2N solver.
Only contains the gamma parameters to preserve global convergence guarantees.
"""
struct R2NArm{T}
    γ1::T
    γ2::T
    γ3::T
end

"""
    R2NContextualSolver

Maintains the standard R2N memory allocations while adding Contextual UCB Bandit 
trackers (N counts, Q values), a history window for state aggregation, and learning rates.
"""
mutable struct R2NContextualSolver{T, V, Sub <: AbstractR2NSubsolver{T}, M <: AbstractNLPModel{T, V}} <: AbstractOptimizationSolver
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
    
    # Contextual Bandit State
    arms::Vector{R2NArm{T}}
    window_size::Int
    history::Vector{Int} # Stores 0 (Fail), 1 (Success), or 2 (Very Success)
    N::Matrix{Int}       # Play counts [num_states, num_arms]
    Q::Matrix{T}         # Q-values [num_states, num_arms]
    ucb_c::T             # Exploration constant
    alpha::T             # Recency learning rate
end

function R2NContextualSolver(
    nlp::AbstractNLPModel{T, V},
    arms::Vector{R2NArm{T}};
    window_size::Int = 5,
    ucb_c::T = T(0.1),
    alpha::T = T(0.2),
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
    
    @assert length(arms) > 0 "Must provide at least one R2NArm"
    @assert window_size > 0 "Window size must be strictly positive"

    # Baseline static parameters (θ and η)
    params = R2NParameterSet(
        nlp; δ1 = δ1, σmin = σmin, non_mono_size = non_mono_size,
        ls_c = ls_c, ls_increase = ls_increase, ls_decrease = ls_decrease,
        ls_min_alpha = ls_min_alpha, ls_max_alpha = ls_max_alpha,
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

    # Bandit Initialization
    num_arms = length(arms)
    num_states = (2 * window_size) + 1 # Max score is 2 * window_size, plus 1 for the zero index
    history = zeros(Int, window_size)
    N = zeros(Int, num_states, num_arms)
    Q = zeros(T, num_states, num_arms)

    return R2NContextualSolver{T, V, typeof(sub_instance), typeof(nlp)}(
        x, xt, gx, rhs, y, Hs, s, scp, obj_vec, sub_instance, h, subtol, σ, params,
        arms, window_size, history, N, Q, ucb_c, alpha
    )
end

function SolverCore.reset!(solver::R2NContextualSolver{T}) where {T}
    fill!(solver.obj_vec, typemin(T))
    fill!(solver.history, 0)
    fill!(solver.N, 0)
    fill!(solver.Q, zero(T))
    if solver.subsolver isa KrylovR2NSubsolver
        LinearOperators.reset!(solver.subsolver.H)
    end
    return solver
end

function get_current_state(solver::R2NContextualSolver)
    return sum(solver.history) + 1
end

function R2NMAB(
    nlp::AbstractNLPModel{T, V},
    arms::Vector{R2NArm{T}};
    kwargs...
) where {T, V}
    solver = R2NContextualSolver(nlp, arms; kwargs...)
    return solve!(solver, nlp; kwargs...)
end

function SolverCore.solve!(
    solver::R2NContextualSolver{T, V},
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
    
    # Static parameters
    η1 = value(params.η1)
    η2 = value(params.η2)
    θ1 = value(params.θ1)
    θ2 = value(params.θ2)
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
    
    done = stats.status != :unknown
    ft = f0
    step_accepted = false
    subiter = 0
    is_npc_ag_step = false

    while !done
        # ==========================================================
        # 1. CONTEXTUAL BANDIT ACTION SELECTION (UCB)
        # ==========================================================
        current_state = get_current_state(solver)
        
        unplayed_idx = findfirst(==(0), solver.N[current_state, :])
        i_k = if unplayed_idx !== nothing
            unplayed_idx # Pure exploration for this specific state
        else
            total_plays = sum(solver.N[current_state, :])
            argmax(solver.Q[current_state, :] .+ solver.ucb_c .* sqrt.(log(T(total_plays)) ./ solver.N[current_state, :]))
        end

        # Extract dynamic gamma parameters
        active_arm = solver.arms[i_k]
        γ1 = active_arm.γ1
        γ2 = active_arm.γ2
        γ3 = active_arm.γ3

        # ==========================================================
        # 2. CORE R2N STEP
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

        # ==========================================================
        # 3. CONTEXT & REWARD UPDATE
        # ==========================================================
        norm_s = norm(s)
        
        # Calculate Reward
        if step_accepted
            final_reward = norm_s / (one(T) + norm_s)
        else
            final_reward = zero(T)
        end
        
        # Update trackers for the specific state and arm chosen
        solver.N[current_state, i_k] += 1
        solver.Q[current_state, i_k] += solver.alpha * (final_reward - solver.Q[current_state, i_k])

        # Shift history queue and push new outcome
        popfirst!(solver.history)
        if !step_accepted
            push!(solver.history, 0)
        elseif ρk >= η2
            push!(solver.history, 2)
        else
            push!(solver.history, 1)
        end

        # ==========================================================
        # Loop Maintenance
        # ==========================================================
        set_iter!(stats, stats.iter + 1)
        set_time!(stats, time() - start_time)

        subtol = max(rtol, min(T(0.1), √norm_∇fk, T(0.9) * subtol))
        set_dual_residual!(stats, norm_∇fk)

        solver.σ = σk
        solver.subtol = subtol
        callback(nlp, solver, stats)

        optimal = norm_∇fk ≤ ϵ
        if stats.status == :user
            done = true
        else
            set_status!(
                stats,
                get_status(nlp, elapsed_time=stats.elapsed_time, optimal=optimal, unbounded=unbounded,
                           max_eval=max_eval, iter=stats.iter, max_iter=max_iter, max_time=max_time)
            )
            done = stats.status != :unknown
        end
    end

    set_solution!(stats, x)
    return stats
end