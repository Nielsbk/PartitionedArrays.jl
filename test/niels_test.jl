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


# GPU -> CPU (CuSparseMatrixCSC -> SparseMatrixCSC)
Adapt.adapt_structure(::Type{Array}, A::CUDA.CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(
    size(A)...,
    collect(A.colPtr),
    collect(A.rowVal),
    collect(A.nzVal),
)


# function cache_to_gpu(cache)

#     graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse = cache

#     V_snd_buf = Adapt.adapt(CuArray,V_snd_buf)
#     V_rcv_buf = Adapt.adapt(CuArray,V_rcv_buf)
#     perm_snd = Adapt.adapt(CuArray,perm_snd)
#     change_snd = Adapt.adapt(CuArray,change_snd)
#     change_sparse = Adapt.adapt(CuArray,change_sparse)
#     perm_sparse = Adapt.adapt(CuArray,perm_sparse)

#     cache = graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse
#     return cache
# end

function time(distribute,n,f,nruns,type)

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    parts_per_dir = (size,)
    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    timing = distribute([[] for i in 1:size ])

    nodes_per_dir = map(i->n,parts_per_dir)
    args = f(nodes_per_dir,parts_per_dir,ranks)

    _,_,V,_,_ = args

    map(V) do val
        if rank == 0
            println(length(val))
        end
    end
    A, cache = PartitionedArrays.psparse_yung_sheng!(sparse,args...) |> fetch

    if type == "cpu"
        t = zeros(nruns)
        @time PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
        for irun in 1:nruns
            t[irun] =  @elapsed PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
        end
        ts_in_main = PartitionedArrays.gather(map(p->t,ranks))
        return ts_in_main
    end


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

    t = zeros(nruns)
    @time PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait
    for irun in 1:nruns
        t[irun] =  @elapsed PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait
    end
    ts_in_main = PartitionedArrays.gather(map(p->t,ranks))
    ts_in_main

    # map(timing)
    # @show A
    # if rank == 1
    #     println("cpu works")
    # end
    # @time PartitionedArrays.psparse_yung_sheng_gpu!(new_A,new_V,new_cache) |> wait
    
    # @time PartitionedArrays.psparse_yung_sheng_gpu!(new_A,new_V,new_cache) |> wait
    # @time PartitionedArrays.psparse_yung_sheng_gpu!(new_A,new_V,new_cache) |> wait


end

# function add_to_df(df,timings,n,f,nruns,type)

function experiment(distribute)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    nruns = 5

    if rank == 0
        df = DataFrame(nodes_per_dir=Int[],sparse_func=String[],nruns=Int[],type=String[], times = PartitionedArrays.JaggedArray{Float64,Int32}[])
    end

    for type in ["cpu","gpu"]
        for n in [10000,100000,1000000,10000000,100000000,200000000]
            params = (n,PartitionedArrays.laplacian_fdm,nruns, type)
            timings = time(distribute,params...)
            PartitionedArrays.map_main(timings) do timing
                push!(df,(n,"laplacian_fdm",nruns, type,timing))
            end
        end
    end

    if rank == 0
        filename = "data_$(size)_nodes.json"
        open(filename,"w") do io
            JSON3.write(io,Tables.columntable(df))
        end
    end

end

PartitionedArrays.with_mpi(experiment)


# function main(distribute)

#     #println(1)
#     np = 4
#     rank = distribute(LinearIndices((np,)))

#     #a = distribute([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])

#     #a = Adapt.adapt(FakeCuVector,a)

#     #println(typeof(a))

#     n = 10
#     parts = rank
#     row_partition = PartitionedArrays.uniform_partition(parts,n)
#     col_partition = row_partition

#     I,J,V = map(parts) do part
#         if part == 1
#             [1,2,1,2,2], [2,6,1,2,1], [1.0,2.0,30.0,10.0,1.0]
#         elseif part == 2
#             [3,3,4,6], [3,9,4,2], [10.0,2.0,30.0,2.0]
#         elseif part == 3
#             [5,5,6,6,6,7], [5,6,2,5,6,7], [10.0,2.0,0.0,0.0,30.0,1.0]
#         else
#             [9,9,8,10,6], [9,3,8,10,5], [10.0,2.0,30.0,50.0,2.0]
#         end
#     end |> PartitionedArrays.tuple_of_arrays

#     copy_I = deepcopy(I)
#     copy_J = deepcopy(J)
#     copy_V = deepcopy(V)
#     A, cache = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch

#     new_A = deepcopy(A)
#     new_cache = deepcopy(cache)

#     comm = MPI.COMM_WORLD
#     # if MPI.Comm_rank(comm) == 1
#     #     println(typeof(new_A))
#     # end

#     new_A = Adapt.adapt(CuArray,new_A)
#     @show PartitionedArrays.local_values(new_A)
#     graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse = new_cache

#     V_snd_buf = Adapt.adapt(CuArray,V_snd_buf)
#     V_rcv_buf = Adapt.adapt(CuArray,V_rcv_buf)
#     perm_snd = Adapt.adapt(CuArray,perm_snd)
#     change_snd = Adapt.adapt(CuArray,change_snd)
#     change_sparse = Adapt.adapt(CuArray,change_sparse)
#     perm_sparse = Adapt.adapt(CuArray,perm_sparse)

#     new_cache = graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse
#     copy_V = deepcopy(V)
#     copy_V = Adapt.adapt(CuArray,copy_V)
#     PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
#     println("cpu works i guess")
#     PartitionedArrays.psparse_yung_sheng_gpu!(new_A,copy_V,new_cache) |> wait



#     new_A = Adapt.adapt(Array,new_A)

#     # if MPI.Comm_rank(comm) == 1
#         # PartitionedArrays.centralize(new_A) |> display
#         # # println("\n\n\n")
#         # PartitionedArrays.centralize(A) |> display
#     # end
#     println(PartitionedArrays.centralize(new_A) == PartitionedArrays.centralize(A))
#     @test PartitionedArrays.centralize(new_A) == PartitionedArrays.centralize(A)
#     # PartitionedArrays.centralize(A) |> display
#     # @show PartitionedArrays.local_values(A)
#     # println("____________________________________________________________")
#     # @assert PartitionedArrays.local_values(new_A) == PartitionedArrays.local_values(A)

#     # @show PartitionedArrays.local_values(A)
#     # println(PartitionedArrays.local_values(A))
#     # map(PartitionedArrays.local_values(A)) do val
#     #     println(val)
#     # end
#     # map(PartitionedArrays.local_values(new_A)) do val
#     #     println(val)
#     # end
#     # @show PartitionedArrays.local_values(new_A)








#     # @assert PartitionedArrays.centralize(PartitionedArrays.local_values(new_A)) == PartitionedArrays.centralize(PartitionedArrays.local_values(A))
#     # map(PartitionedArrays.local_values(new_A),PartitionedArrays.local_values(A)) do a,b
#     #     println("ites")
#     #     @assert a == b
#     # end
#     # if MPI.Comm_rank(comm) == 1
#     #     println(typeof(new_A))
#     # end
#     # send = new_cache.V_snd_buf
#     # # println(typeof(send))
#     # if MPI.Comm_rank(comm) == 1
#     #     println(typeof(send))
#     # end
#     # send = map(send) do val
#     #     println(val)
#     # end
#     # send = Adapt.adapt(CuArray,send)
#     # send = map(send) do val
#     #     CUDA.@allowscalar @show Array(val)
#     # end
#     # # println(typeof(send))
#     # if MPI.Comm_rank(comm) == 1
#     #     println(typeof(send))
#     # end
#     # map(new_A.matrix_partition) do values
#     #     println(typeof(values.blocks))
#     #     println(typeof(values.blocks.own_own))
#     #     @show values
#     # end
#     # println(typeof(new_A.row_partition))
#     # println(typeof(new_A.col_partition))
#     # println(typeof(new_cache))
#     # println(new_cache)

#     #copy_V = deepcopy(V)
#     # PartitionedArrays.psparse_yung_sheng!(new_A, copy_V, new_cache) |> wait
#     # @time PartitionedArrays.psparse_yung_sheng!(new_A, copy_V, new_cache) |> wait
#     # A,cache = psparse(I,J,V,row_partition,col_partition,split_format=false,reuse=true) |> fetch
#     # psparse!(A,V,cache) |> wait
# end

# PartitionedArrays.with_mpi(main)
# PartitionedArrays.with_mpi(time)
