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

function fast_sparse_eq(A::SparseMatrixCSC, B::SparseMatrixCSC)
    size(A) == size(B) &&
    A.colptr == B.colptr &&
    A.rowval == B.rowval &&
    A.nzval == B.nzval
end

# GPU -> CPU (CuSparseMatrixCSC -> SparseMatrixCSC)
Adapt.adapt_structure(::Type{Array}, A::CUDA.CUSPARSE.CuSparseMatrixCSC) = SparseMatrixCSC(
    size(A)...,
    convert(Vector{Int}, collect(A.colPtr)),
    convert(Vector{Int}, collect(A.rowVal)),
    collect(A.nzVal),
)

Adapt.adapt_structure(::Type{CuArray}, A::SparseMatrixCSC) = CUDA.CUSPARSE.CuSparseMatrixCSC(
    size(A)...,
    CuArray(Int64.(A.colptr)),
    CuArray(Int64.(A.rowval)),
    CuArray(A.nzval),
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

function test_sizes(distribute)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    parts_per_dir = (1,1,1)
    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))

    nodes_per_dir = map(i->i * 20,parts_per_dir)
    args = PartitionedArrays.laplacian_fdm(nodes_per_dir,parts_per_dir,ranks)

    _,_,V,_,_ = args
    V_len = length(V)
    map(V) do val
        if rank == 0
            println(length(val))
        end
    end
    # check if it fits gpu
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


end


function profile(distribute)
    function sparse_matrix!(A, V, K; reset=true)
        if reset
            CUDA.fill!(A.nzVal, 0)  # Reset nonzero values on GPU
        end
        
        function kernel_update!(A_nz, V, K, N)
            i = (blockIdx().x - 1) * blockDim().x + threadIdx().x
            if i ≤ N && K[i] > 0 && i > 0
                CUDA.@atomic A_nz[K[i]] += V[i]  # Update nonzero elements
            end
            return
        end
    
        A_nz = A.nzVal  # Get the nonzero values array
        N = length(V)
        if N == 0
            println("empty sparse_matrix warning")
            return A
        end
        threads = 256
        blocks = cld(N, threads)
    
        CUDA.@cuda threads=threads blocks=blocks kernel_update!(A_nz, V, K, N)
    
        return A
    end

    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    parts_per_dir = (size,)
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
    p = prod(parts_per_dir)
    ranks = distribute(LinearIndices((p,)))
    timing = distribute([[] for i in 1:size ])

    nodes_per_dir = map(i->80,parts_per_dir)
    args = PartitionedArrays.laplacian_fdm(nodes_per_dir,parts_per_dir,ranks)

    _,_,V,_,_ = args
    V_len = length(V)
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

    # map(sparse_matrix!(A,V,K))
    # PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait

    
    # p = CUDA.@profile PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait
    # open("profile.txt", "a") do io
    #     println("lenght of V: $V_len")
    #     println("---------------------------------------")
    #     println(io, p)
    #     println("---------------------------------------")
    # end
    # p = CUDA.@profile PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait
    # open("profile.txt", "a") do io
    #     println(io, p)
    # end
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
function time(distribute,n,f,nruns,type)

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
        PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
        for irun in 1:nruns
            t[irun] =  @elapsed PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
        end
        ts_in_main = PartitionedArrays.gather(map(p->t,ranks))
        return ts_in_main, PartitionedArrays.gather(map(p->length(p),V)),parts_per_dir, nodes_per_axis,gpus_per_axis
    end

    PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
    A_test = deepcopy(A)
    cache_test = deepcopy(cache)

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

    map(V) do val
        if rank == 0
            println(typeof(A_test))
            println(typeof(A))
        end
    end
    t = zeros(nruns)
    PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait
    for irun in 1:nruns
        t[irun] =  @elapsed CUDA.@sync PartitionedArrays.psparse_yung_sheng_gpu!(A,V,cache) |> wait
    end
    ts_in_main = PartitionedArrays.gather(map(p->t,ranks))

    A = Adapt.adapt(Array,A)

    if rank == 0
        try 
            CUDA.pool_status()
            println
            @test fast_sparse_eq(PartitionedArrays.centralize(A), PartitionedArrays.centralize(A_test))
            println("passed with size $(n)")
        catch
            println("failed with size $(n)")
        end
    end
    return ts_in_main, PartitionedArrays.gather(map(p->length(p),V)),parts_per_dir, nodes_per_axis,gpus_per_axis

end

# function add_to_df(df,timings,n,f,nruns,type)

function experiment(distribute)
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)
    shared_comm = MPI.Comm_split_type(comm, MPI.COMM_TYPE_SHARED, 0)

    # Get the local rank and local size
    local_size = MPI.Comm_size(shared_comm)
    nruns = 2
    filename="strongscaling_sync_$(size)_$(local_size)_snellius.json"

    df = DataFrame()
    if rank == 0
        # try
        #     json_data = JSON3.read(open(filename, "r"))
        #     df = DataFrame(json_data)
        # catch
            df = DataFrame(nodes_per_dir=Int[],sparse_func=String[],nruns=Int[],type=String[], times = PartitionedArrays.JaggedArray{Float64,Int32}[],workers=Int[],nzc=Int[],distribution=Tuple{Int64, Int64, Int64}[],nodes_per_axis=Tuple{Int64, Int64, Int64}[],gpus_per_axis=Tuple{Int64, Int64, Int64}[])
        # end
    end

    for type in ["gpu"]
        for n in [20,50,100,150,200,300,400]
            params = (n,PartitionedArrays.laplacian_fdm,nruns, type)
            timings,nnz,parts_per_dir, nodes_per_axis,gpus_per_axis= time(distribute,params...)
            nz = 0
            PartitionedArrays.map_main(nnz) do i
                nz = i
            end
            PartitionedArrays.map_main(timings) do timing
                push!(df,(n,"laplacian_fdm",nruns, type,timing,size,size*(nz[1]),parts_per_dir, nodes_per_axis,gpus_per_axis))
            end
        end
    end

    # if rank == 0
    #     open(filename,"w") do io
    #         JSON3.write(io,Tables.columntable(df))
    #     end
    # end

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