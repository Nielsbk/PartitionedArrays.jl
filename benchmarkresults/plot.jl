import PartitionedArrays as PA
using DataFrames
using JSON
using Glob
using Tables
using Statistics
using Plots

function wall_times(timings)
    cleaned = [Vector{Float64}(row) for row in timings]
    times= []
    for i in 1:length(cleaned[1])
        col = [row[i] for row in cleaned]  # Get the i-th element from each vector
        # Example: check if all are equal
        push!(times,maximum(col))
    end
    return times
end

function get_dataframe(regex_json)
    json_files = glob(regex_json, "./")

    println(json_files)


    dfs = DataFrame[]


    for file in json_files
        data = JSON.read(file,String)    
        df = DataFrame(JSON.parse(data)) 
        df.num_nodes = fill(parse(Int, match(r"\d+", file).match), nrow(df))
        push!(dfs, df)                               
    end

    df = vcat(dfs...)

    df.wall_times = wall_times.(df.times)

    df.median_time = median.(df.wall_times)
    df.std_time = std.(df.wall_times)
    df.mean_time = mean.(df.wall_times)
    df.best_time = minimum.(df.wall_times)

    return df
end

function speedup_experiment()
    df = get_dataframe("*_nodes.json")

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

function speedup_experiment_consistent()
    df = get_dataframe("*_per_node.json")

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

function speedup_experiment_nodes()
    df = get_dataframe("*nodes.json")

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

function weak_scaling(files)
    df = get_dataframe(files)
    num_nodes = unique(df.num_nodes)
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
                title = " Strong scaling",
                linewidth = 2,
                legend = :outertopright
            )
        end
        plot!(num_nodes, speedup, label = "Problem size: $n")
        
    end
    savefig("weak_scaling.png")
    
end
# speedup_experiment()
# speedup_experiment_consistent()
# speedup_experiment_nodes()
weak_scaling("*per_node.json")
strong_scaling("*nodes.json")

