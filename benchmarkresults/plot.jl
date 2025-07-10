import PartitionedArrays as PA
using DataFrames
using JSON
using Glob
using Tables
using Statistics
using Plots
using JSON3

markers = filter((m->begin
                m in Plots.supported_markers()
            end), Plots._shape_keys)
markers = permutedims(markers)


function wall_times(timings)
    # cleaned = [Vector{Float64}(row) for row in timings]
    cleaned = [row for row in timings]
    times= []
    for i in 1:length(cleaned[1])
        col = [row[i] for row in cleaned]  # Get the i-th element from each vector

        push!(times,maximum(col))
    end
    return times
end

function flatten_dict(d::Dict{String, @NamedTuple{min::Float64, max::Float64, avg::Float64}})
    return (; (key * "_" * String(k) => v for (key, nt) in d for (k, v) in pairs(nt))...)
end

function combine_txt(start)
    folder = "./snellius/"  # replace with your actual folder path

    files_starting_with_x = filter(f -> startswith(f, start), readdir(folder))

    files = filter(f -> isfile(joinpath(folder, f)), files_starting_with_x)

    dfs = []
    for f in files 
        matches = collect(eachmatch(r"\d+", f))

        # Get the first and second integers (as Int)
        first_int = parse(Int, matches[1].match)
        second_int = parse(Int, matches[2].match)
        txt = read(folder*f, String)
        dict = eval(Meta.parse(txt))
        df = DataFrame(dict)
        df.num_workers = fill(first_int,nrow(df))
        df.n = fill(second_int,nrow(df))
        push!(dfs, df)
    end
    df = vcat(dfs...)
    open(start*".json","w") do io
            JSON3.write(io,Tables.columntable(df))
    end
end

function combine_json(start)
    folder = "./snellius/"  # replace with your actual folder path

    # println(readdir(folder))
    # Get all files starting with "x"
    files_starting_with_x = filter(f -> startswith(f, start), readdir(folder))

    # (Optional) Get only files (not directories)
    json_files = filter(f -> isfile(joinpath(folder, f)), files_starting_with_x)

    println(json_files)
    dfs = DataFrame[]
     for file in json_files
        data = JSON.read(folder*file,String)    
        df = DataFrame(JSON.parse(data)) 
        try
            df.num_nodes = fill(prod(df.nodes_per_axis[1]), nrow(df))
            df.num_gpu_per_node = fill(prod(df.gpus_per_axis[1]), nrow(df))
        catch e 
        
        end
        push!(dfs, df)                               
    end

    df = vcat(dfs...)
    
    try
        df.wall_times = wall_times.(df.times)
        df.median_time = median.(df.wall_times)
        df.std_time = std.(df.wall_times)
        df.mean_time = mean.(df.wall_times)
        df.best_time = minimum.(df.wall_times)
    catch e
    end
    open(start*".json","w") do io
            JSON3.write(io,Tables.columntable(df))
    end

end

function read_dfjson(filename)
    json_data = JSON3.read(read(filename, String))
    df = DataFrame(json_data)
    df
end

function write_dfjson(df,filename)
        open(filename,"w") do io
            JSON3.write(io,Tables.columntable(df))
    end
end
function get_dataframe(regex_json)
    json_files = glob(regex_json, "./")

    println(json_files)


    dfs = DataFrame[]


    for file in json_files
        data = JSON.read(file,String)    
        df = DataFrame(JSON.parse(data)) 
        try
            df.num_nodes = fill(parse(Int, match(r"\d+", file).match), nrow(df))
        catch e 

            transform!(df, :distribution => ByRow(prod) => :num_workers)
        end
        push!(dfs, df)                               
    end

    df = vcat(dfs...)

    try
        df.wall_times = wall_times.(df.times)
        df.median_time = median.(df.wall_times)
        df.std_time = std.(df.wall_times)
        df.mean_time = mean.(df.wall_times)
        df.best_time = minimum.(df.wall_times)
    catch e
        
    end
   

    return df
end

function speedup_experiment(files)
    df = get_dataframe(files)

    nodes = unique(df.num_nodes)

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_cpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "cpu") , :]
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort(df_gpu,[:num_nodes])
        sort(df_cpu,[:num_nodes])
        speedup = df_cpu[!,"median_time"]  ./ df_gpu[!,"median_time"]

        if i == 1
            plot(
                nodes, speedup,
                label = "Problem size: $n",
                xlabel = "Number of Nodes",
                ylabel = "Speedup",
                title = "Parallel Speedup vs. Number of Nodes with different problem size",
                # marker = :circle,
                linewidth = 2,
                legend = :outertopright
            )

        else
            plot!(nodes, speedup, label = "Problem size: $n")
        end

    end
    savefig("speedup_plot.png")
end

function speedup_experiment_consistent(files)
    df = get_dataframe(files)

    nodes = unique(df.num_nodes)

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_cpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "cpu") , :]
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort(df_gpu,[:num_nodes])
        sort(df_cpu,[:num_nodes])
        speedup = df_cpu[!,"median_time"]  ./ df_gpu[!,"median_time"]

        if i == 1
            plot(
                nodes, speedup,
                label = "Problem size: $n",
                xlabel = "Number of Nodes",
                ylabel = "Speedup",
                title = " Speedup cpu vs. gpu  Number of Nodes with different problem size consistent on each node",
                # marker = :circle,
                linewidth = 2,
                legend = :outertopright
            )

        else
            plot!(nodes, speedup, label = "Problem size: $n")
        end

    end
    savefig("speedup_consistent_work_plot.png")
end

function speedup_experiment_nodes(files)
    df = get_dataframe(files)

    nodes_per_dir = unique(df.nodes_per_dir)

    for (i,n) in enumerate(unique(df.num_nodes))
        df_cpu = df[(df.num_nodes .== n) .&& (df.type .== "cpu") , :]
        df_gpu = df[(df.num_nodes .== n) .&& (df.type .== "gpu") , :]
        sort(df_gpu,[:nodes_per_dir])
        sort(df_cpu,[:nodes_per_dir])
        speedup = df_cpu[!,"median_time"]  ./ df_gpu[!,"median_time"]

        if i == 1
            plot(
                nodes_per_dir, speedup,
                label = "number of nodes: $n",
                xlabel = "problem size",
                ylabel = "Speedup",
                title = " Speedup gpu/cpu vs problem size with different number of nodes",
                # marker = :circle,
                linewidth = 2,
                legend = :outertopright
            )

        else
            plot!(nodes_per_dir, speedup, label = "Problem size: $n")
        end

    end
    savefig("speedup_nodes_work_plot.png")
end


function strong_scaling(files)
    df = get_dataframe(files)
    num_nodes = unique(df.num_nodes)
    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort(df_gpu,[:num_nodes])
        base = df_gpu[!,"median_time"][1]

 
        # println(base)
        speedup =   base ./ df_gpu[!,"median_time"]
        if i == 1
            plot(
                num_nodes, num_nodes,
                label = "Ideal",
                linecolor=:black,
                linestyle=:dash,
                xlabel = "workers",
                ylabel = "Speedup",
                title = " Strong scaling",
                linewidth = 2,
                legend = :outertopright
            )
        end
        plot!(num_nodes, speedup, label = "Problem size: $n")
        
    end
    savefig("strong_scaling.png")
    
end

function weak_scaling(files,filename)
    df = get_dataframe(files)
    num_nodes = sort(unique(df.num_nodes))
    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort(df_gpu,[:num_nodes])
        base = df_gpu[!,"median_time"][1]
        speedup =   base ./ df_gpu[!,"median_time"]
        if i == 1
            plot(
                num_nodes, [1.0 for i in num_nodes],
                label = "Ideal",
                linecolor=:black,
                linestyle=:dash,
                xlabel = "workers",
                ylabel = "Speedup",
                title = " Weak scaling",
                linewidth = 2,
                legend = :outertopright
            )
        end
        plot!(num_nodes, speedup, label = "Problem size: $n")
        
    end
    savefig(filename)
    
end

function weak_scaling_snellius(files,filename)
    df = get_dataframe(files)
    num_workers = sort(unique(df.num_workers))
    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort(df_gpu,[:num_workers])
        base = df_gpu[!,"median_time"][1]
        speedup =   base ./ df_gpu[!,"median_time"]
        if i == 1
            plot(
                num_workers, [1.0 for i in num_workers],
                label = "Ideal",
                linecolor=:black,
                linestyle=:dash,
                xlabel = "workers",
                ylabel = "Speedup",
                title = " Weak scaling",
                linewidth = 2,
                marker = :circle,
                legend = :outertopright
            )
        end
        plot!(num_workers, speedup, label = "Problem size: $n")
        
    end
    savefig(filename)
    
end
















function strong_scaling_snellius_experiment(df,filename)
    df = df[df.nodes_per_dir .< 350 , :]
    num_workers = sort(unique(df.workers))
    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort!(df_gpu,[:workers])
        base = df_gpu[!,"best_time"][1]

        speedup =   base ./ df_gpu[!,"best_time"]
        if i == 1
            plot(
                num_workers, num_workers,
                label = "Ideal",
                # yscale = :log10,

                xticks=2:2:20,
                linecolor=:black,
                linestyle=:dash,
                xlabel = "GPUs",
                ylabel = "Speedup",
                title = " Strong scaling",
                linewidth = 2,
                legend = :outertopright
            )
        end
        plot!(num_workers, speedup, label = "Problem size: $(n)³",marker=markers[i])
    
    end
    savefig("speedup_"*filename)

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_cpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "cpu") , :]
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]

        sort!(df_gpu,[:workers])
        sort!(df_cpu,[:workers])
        speedup = df_cpu[!,"best_time"]  ./ df_gpu[!,"best_time"]

        if i == 1
            plot(
                num_workers, speedup,
                label = "Problem size: $(n)³",
                yscale = :log10,

                xticks=2:2:20,
                xlabel = "Workers",
                ylabel = "Speedup (GPU / CPU)",
                title = " Strong scaling speedup GPU vs CPU",
                marker=markers[i],
                legend = :outertopright
            )
        
        else
            plot!(num_workers, speedup, label = "Problem size: $(n)³ ",marker=markers[i])
        end
    end
    savefig("gpu_vs_cpu_"*filename)

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        
        sort!(df_gpu,[:workers])
        time = df_gpu[!,"best_time"]

        if i == 1
            plot(
                num_workers, time,
                xlabel = "GPUs",
                ylabel = "Time(s)",
                yscale = :log10,
                # xscale = :log10,
                xticks=2:2:20,
                title = " Strong scaling wall times",
                linewidth = 2,
                legend = :outertopright,
                label = "Problem size: $n",
                marker=markers[i]
            )
        
        else
            plot!(num_workers, time, label = "Problem size: $(n)³",marker=markers[i])
        end
    end
    savefig("walltimes_"*filename)
    
end


function weak_scaling_snellius_experiment(df,filename)
    num_workers = sort(unique(df.workers))

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        sort!(df_gpu,[:workers])
        base = df_gpu[!,"best_time"][1]
        speedup =   base ./ df_gpu[!,"best_time"]
        if i == 1
            plot(
                num_workers, [1.0 for i in num_workers],
                label = "Ideal",
                linecolor=:black,
                linestyle=:dash,
                xticks=2:2:20,
                xlabel = "GPUs",
                ylabel = "efficiency",
                title = " Weak scaling efficiency",
                linewidth = 2,
                legend = :outertopright
            )
        end
        plot!(num_workers, speedup, label = "Local problem size: $(n)³", marker=markers[i])
        
    end
    savefig("speedup_"*filename)

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_cpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "cpu") , :]
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]

        sort!(df_gpu,[:workers])
        sort!(df_cpu,[:workers])
        speedup = df_cpu[!,"best_time"]  ./ df_gpu[!,"best_time"]

        if i == 1
            plot(
                num_workers, speedup,
                label = "Problem size: $(n)³",
                xlabel = "GPUs/CPUs",
                # yscale = :log2,
                # xscale = :log2,
                xticks=2:2:20,
                ylabel = "Speedup",
                title = " Weak scaling speedup GPU vs CPU",
                marker=markers[i],
                legend = :outertopright
            )
        
        else
            plot!(num_workers, speedup, label = "Local problem size: $(n)³",marker=markers[i])
        end
    end
    savefig("gpu_vs_cpu_"*filename)

    for (i,n) in enumerate(unique(df.nodes_per_dir))
        df_gpu = df[(df.nodes_per_dir .== n) .&& (df.type .== "gpu") , :]
        
        sort!(df_gpu,[:workers])
        time = df_gpu[!,"best_time"]

        if i == 1
            plot(
                num_workers, time,
                xlabel = "GPUs",
                ylabel = "Time(s)",
                yscale = :log10,
                # xscale = :log2,
                xticks=2:2:20,
                title = " Weak scaling wall times",
                linewidth = 2,
                legend = :outertopright,
                label = "Local problem size: $(n)³",
                marker=markers[i]
            )
            continue
        end
        plot!(num_workers, time, label = "Local problem size: $(n)³",marker=markers[i])
    end
    savefig("walltimes_"*filename)

end








function time_experiment(df,filename,type)
    num_workers = sort(unique(df.num_workers))
    problem_sizes = sort(unique(df.n))
    # sleutels = ["exchange","partition_and_prepare_snd_buf","store_recv_data"]
    sleutels = names(df)[1:end-2]

    
    for n in problem_sizes
        df_temp = df[(df.n .== n) , :]
        sort!(df_temp,[:num_workers])
        for (i,key) in enumerate(sleutels)
            # Extract avg, min, max values for the chosen key from each dict
            # avg_vals = [d[:avg] for d in df_temp[!,key]]
            min_vals = [d[:min] for d in df_temp[!,key]]
            # max_vals = [d[:max] for d in df_temp[!,key]]

            # Calculate lower and upper error bars
            # yerr_lower = avg_vals .- min_vals
            # yerr_upper = max_vals .- avg_vals
            # xticks = minimum(num_workers):maximum(num_workers)
            # Plot line with asymmetric error bars
            if i == 1
                plot(num_workers, min_vals,
                    # yerror = (yerr_lower, yerr_upper),
                    xlabel = "GPUs",
                    ylabel = "Time(s)",
                    title = "$(type) timings with problem size $(n)",
                    legend = :outertopright,
                    label = key,
                    # xticks=xticks,
                    yscale = :log10,
                    yticks = (10.0 .^ (-10:-1), ["10^{-$(i)}" for i in 10:-1:1]),
                    yguide = "Log-scaled values",
                    # xscale = :log10,
                    xticks=2:2:20,
                    lw = 2,
                    marker = markers[i])
            else
                plot!(num_workers, min_vals,
                    # yerror = (yerr_lower, yerr_upper),
                    label=key,
                    marker = markers[i]
                )
            end
        end
        savefig(filename*"_$(n).png")
    end

    # for gpus in [1]
    #     df_temp = df[(df.num_workers .== gpus) , :]
    #     sort!(df_temp,[:n])
    #     for (i,key) in enumerate(sleutels)
    #         avg_vals = [d[:avg] for d in df_temp[!,key]]
    #         if i == 1
    #             plot(problem_sizes, avg_vals,

    #                 xlabel = "Problem size",
    #                 ylabel = "Time(s)",
    #                 title = "$(type) time distribution of different parts",
    #                 legend = :outertopright,
    #                 label = key,
    #                 lw = 2,
    #                 marker = markers[i])
    #         else
    #             plot!(problem_sizes, avg_vals,
    #                 label=key,
    #                 marker = markers[i]
    #             )
    #         end
    #     end
    #     savefig(filename*"size_vs_time_$(gpus).png")
    # end
    
end

function perc_wall_time()
    data = JSON.read("times.json",String) 
    df = DataFrame(JSON.parse(data)) 
    num_workers = sort(unique(df.num_workers))
    sort(df,[:num_workers])
    keys = ["exchange","split_and_compress","partition_and_prepare_snd_buf","store_recv_data"]

    for (i,key) in enumerate(keys)
        # Extract avg, min, max values for the chosen key from each dict
        avg_vals = [d["avg"] for d in df[!,key]]
        min_vals = [d["min"] for d in df[!,key]]
        max_vals = [d["max"] for d in df[!,key]]

        # Calculate lower and upper error bars
        yerr_lower = avg_vals .- min_vals
        yerr_upper = max_vals .- avg_vals

        # Plot line with asymmetric error bars
        if i == 1
            plot(num_workers, avg_vals,
                yerror = (yerr_lower, yerr_upper),
                xlabel = "workers",
                ylabel = "Time",
                title = "$key timings with error bars",
                legend = :outertopright,
                label = key,
                lw = 2,
                marker = :circle)
        else
            plot!(num_workers, avg_vals,
                yerror = (yerr_lower, yerr_upper),
                label=key
            )
        end
    end
    savefig("timings.png")
end



# speedup_experiment("*strong.json")
# speedup_experiment_consistent("*weak.json")
# speedup_experiment_nodes("*strong.json")
# weak_scaling("*weak.json")
# strong_scaling("*strong.json")

