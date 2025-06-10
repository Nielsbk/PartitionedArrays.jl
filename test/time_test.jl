import PartitionedArrays
using Test
using LinearAlgebra
using Random
using Distances
using SparseArrays
using IterativeSolvers
import Adapt
using CUDA
using MPI
using DataFrames
using JSON3
using Statistics

function average_timings(dicts::Vector{Dict{String, NamedTuple{(:min, :max, :avg), Tuple{Float64, Float64, Float64}}}})
    result = Dict{String, NamedTuple{(:min, :max, :avg), Tuple{Float64, Float64, Float64}}}()

    # Collect all unique keys (phases) from all dicts
    all_keys = Set{String}()
    for d in dicts
        union!(all_keys, keys(d))
    end

    # For each key, gather values and compute averages
    for key in all_keys
        mins = Float64[]
        maxs = Float64[]
        avgs = Float64[]

        for d in dicts
            if haskey(d, key)
                push!(mins, d[key].min)
                push!(maxs, d[key].max)
                push!(avgs, d[key].avg)
            end
        end

        if !isempty(mins)
            result[key] = (
                min = mean(mins),
                max = mean(maxs),
                avg = mean(avgs)
            )
        end
    end

    return result
end

function calc_parts(size,local_size)
    parts_per_dir = (size,)
    parts_per_dir_local = (size,)
    if size == 1
        parts_per_dir = (1,1,1)
    end
    if size == 2
        parts_per_dir = (1,1,2)
    end
    if size == 3
        parts_per_dir = (1,1,3)
    end
    if size == 4
        parts_per_dir = (1,2,2)
    end
    if size == 6
        parts_per_dir = (1,2,3)
    end
    if size == 8
        parts_per_dir = (2,2,2)
    end

    if local_size == 1
        parts_per_dir_local = (1,1,1)
    end
    if local_size == 2
        parts_per_dir_local = (2,1,1)
    end
    if local_size == 3
        parts_per_dir_local = (3,1,1)
    end
    if local_size == 4
        parts_per_dir_local = (2,2,1)
    end

    return parts_per_dir, parts_per_dir_local
end

function experiment(distribute)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    shared_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, 0)

    # Get the local rank and local size
    local_size = MPI.Comm_size(shared_comm)
    num_nodes = size / local_size

    nodes_per_axis, gpus_per_axis = calc_parts(num_nodes,local_size)
    parts_per_dir = Base.broadcast(*, nodes_per_axis, gpus_per_axis)

    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    timing = distribute([[] for i in 1:size ])
    t = PartitionedArrays.PTimer(ranks)

    nodes_per_dir = map(i->i*200,parts_per_dir)
    args = PartitionedArrays.laplacian_fdm(nodes_per_dir,parts_per_dir,ranks)

    _,_,V,_,_ = args
    v_len = length(V)
    map(V) do val
        if rank == 0
            println(length(val))
        end
    end
    A, cache = PartitionedArrays.psparse_yung_sheng!(sparse,args...) |> fetch
    graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse = cache

    V_snd_buf = Adapt.adapt(CuArray,V_snd_buf)
    V_rcv_buf = Adapt.adapt(CuArray,V_rcv_buf)
    perm_snd = Adapt.adapt(CuArray,perm_snd)
    change_snd = Adapt.adapt(CuArray,change_snd)
    change_sparse = Adapt.adapt(CuArray,change_sparse)
    perm_sparse = Adapt.adapt(CuArray,perm_sparse)

    cache = graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse

    # new_cache = cache_to_gpu(new_cache)
    A = Adapt.adapt(CuArray,A)
    V = Adapt.adapt(CuArray,V)
    dicts = Dict{String, NamedTuple{(:min, :max, :avg), Tuple{Float64, Float64, Float64}}}[]

    for i in 1:10
        PartitionedArrays.psparse_yung_sheng_gpu_time!(A,V,cache,t) |> wait
        push!(dicts,PartitionedArrays.statistics(t))
    end
        # A,t = PartitionedArrays.psparse_yung_sheng_gpu_time!(A,V,cache,t)


    dict = average_timings(dicts)
    PartitionedArrays.map_main(ranks) do part
        open("benchmarkresults/times_$(size)_$(local_size)_snellius.txt","w") do io
            println(io,dict)
        end
    end
end
PartitionedArrays.with_mpi(experiment)