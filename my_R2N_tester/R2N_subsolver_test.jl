println("==============================================================")
println("      Exhaustive Testing R2N Combinations                     ")
println("==============================================================")

# 1. Define the Problem
n = 30
nlp = ADNLPModel(
    x -> sum(100 * (x[i + 1] - x[i]^2)^2 + (x[i] - 1)^2 for i = 1:(n - 1)),
    collect(1:n) ./ (n + 1),
    name = "Extended Rosenbrock"
)

# 2. Define Parameter Grids for Combinations
subsolvers = [CGR2NSubsolver, CRR2NSubsolver, MinresR2NSubsolver, MinresQlpR2NSubsolver]

# Automatically append HSL solvers if available
if LIBHSL_isfunctional()
    push!(subsolvers, MA97R2NSubsolver, MA57R2NSubsolver)
end

npc_handlers = [:ag, :sigma, :prev, :cp]
scp_flags    = [true, false]
always_accept_npc_ags = [true, false]
fast_local_convergences = [true, false]

# 3. Storage for Results
passed_runs = []
failed_runs = []

# Calculate total combinations
total_combinations = length(subsolvers) * length(npc_handlers) * length(scp_flags) * length(always_accept_npc_ags) * length(fast_local_convergences)
println("Testing $total_combinations combinations...\n")

# 4. Execution Loop
current_run = 1
for params in Iterators.product(subsolvers, npc_handlers, scp_flags, always_accept_npc_ags, fast_local_convergences)
    sub_type, handler, scp, accept_ag, fast_conv = params
    
    # Create a string identifying this exact setup
    config_name = "Sub: $(string(sub_type)) | NPC: $(handler) | scp: $(scp) | accept_ag: $(accept_ag) | fast_conv: $(fast_conv)"
    
    print("\rProgress: $current_run / $total_combinations combinations tested...")
    
    try
        # Run silently with a smaller max_iter so the mass-test finishes quickly
        stats = R2N(
            nlp; 
            verbose = 0, 
            max_iter = 100, 
            subsolver = sub_type, 
            npc_handler = handler,
            scp_flag = scp,
            always_accept_npc_ag = accept_ag,
            fast_local_convergence = fast_conv
        )
        push!(passed_runs, (config_name, stats.status, stats.iter, stats.objective))
    catch e
        # If it crashes, catch the error and the backtrace so we can inspect it later
        bt = catch_backtrace()
        err_msg = sprint(showerror, e, bt)
        push!(failed_runs, (config_name, err_msg))
    end
    
    current_run += 1
end

println("\n\n==============================================================")
println("                    Testing Summary                           ")
println("==============================================================")
println("Total Tested: $total_combinations")
println("Passed:       $(length(passed_runs))")
println("Failed:       $(length(failed_runs))")
println("==============================================================\n")

# 5. Print Error Report
if !isempty(failed_runs)
    println("### 🚨 ERROR REPORT 🚨 ###\n")
    for (config, err) in failed_runs
        println("❌ CONFIGURATION:")
        println("   ", config)
        println("\n   ERROR DETAILS:")
        
        # Print just the first few lines of the stacktrace to avoid terminal flooding
        err_lines = split(err, '\n')
        for line in err_lines[1:min(12, length(err_lines))]
            println("      ", line)
        end
        println("-" ^ 80)
    end
else
    println("🎉 All configurations ran without throwing exceptions!")
end

# Optional: Print a small sample of passed runs to verify it's working
if !isempty(passed_runs)
    println("\nSample of Passed Runs:")
    @printf("%-100s %-15s %-5s\n", "Configuration", "Status", "Iter")
    println("-" ^ 125)
    for (cfg, st, it, obj) in passed_runs[1:min(5, length(passed_runs))]
        @printf("%-100s %-15s %-5d\n", cfg, st, it)
    end
end

# Clean up CUTEst model memory
finalize(nlp)