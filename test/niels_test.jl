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

function main(distribute)

    #println(1)
    np = 4
    rank = distribute(LinearIndices((np,)))

    #a = distribute([[1,2,3],[1,2,3],[1,2,3],[1,2,3]])

    #a = Adapt.adapt(FakeCuVector,a)

    #println(typeof(a))

    n = 10
    parts = rank
    row_partition = PartitionedArrays.uniform_partition(parts,n)
    col_partition = row_partition

    I,J,V = map(parts) do part
        if part == 1
            [1,2,1,2,2], [2,6,1,2,1], [1.0,2.0,30.0,10.0,1.0]
        elseif part == 2
            [3,3,4,6], [3,9,4,2], [10.0,2.0,30.0,2.0]
        elseif part == 3
            [5,5,6,6,6,7], [5,6,2,5,6,7], [10.0,2.0,0.0,0.0,30.0,1.0]
        else
            [9,9,8,10,6], [9,3,8,10,5], [10.0,2.0,30.0,50.0,2.0]
        end
    end |> PartitionedArrays.tuple_of_arrays

    copy_I = deepcopy(I)
    copy_J = deepcopy(J)
    copy_V = deepcopy(V)
    A, cache = PartitionedArrays.psparse_yung_sheng!(sparse, copy_I, copy_J, copy_V, row_partition, col_partition) |> fetch

    new_A = deepcopy(A)
    new_cache = deepcopy(cache)

    comm = MPI.COMM_WORLD
    # if MPI.Comm_rank(comm) == 1
    #     println(typeof(new_A))
    # end

    new_A = Adapt.adapt(CuArray,new_A)
    graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse = new_cache

    V_snd_buf = Adapt.adapt(CuArray,V_snd_buf)
    V_rcv_buf = Adapt.adapt(CuArray,V_rcv_buf)
    perm_snd = Adapt.adapt(CuArray,perm_snd)
    change_snd = Adapt.adapt(CuArray,change_snd)
    change_sparse = Adapt.adapt(CuArray,change_sparse)
    perm_sparse = Adapt.adapt(CuArray,perm_sparse)

    new_cache = graph, V_snd_buf, V_rcv_buf, hold_data_size, snd_start_idx, change_snd, perm_snd, own_data_size, change_sparse, perm_sparse
    copy_V = deepcopy(V)
    copy_V = Adapt.adapt(CuArray,copy_V)
    PartitionedArrays.psparse_yung_sheng!(A,V,cache) |> wait
    PartitionedArrays.psparse_yung_sheng_gpu!(new_A,copy_V,new_cache) |> wait





    println(typeof(new_A))
    println(typeof(A))
    # if MPI.Comm_rank(comm) == 1
    #     println(typeof(new_A))
    # end
    # send = new_cache.V_snd_buf
    # # println(typeof(send))
    # if MPI.Comm_rank(comm) == 1
    #     println(typeof(send))
    # end
    # send = map(send) do val
    #     println(val)
    # end
    # send = Adapt.adapt(CuArray,send)
    # send = map(send) do val
    #     CUDA.@allowscalar @show Array(val)
    # end
    # # println(typeof(send))
    # if MPI.Comm_rank(comm) == 1
    #     println(typeof(send))
    # end
    # map(new_A.matrix_partition) do values
    #     println(typeof(values.blocks))
    #     println(typeof(values.blocks.own_own))
    #     @show values
    # end
    # println(typeof(new_A.row_partition))
    # println(typeof(new_A.col_partition))
    # println(typeof(new_cache))
    # println(new_cache)

    #copy_V = deepcopy(V)
    # PartitionedArrays.psparse_yung_sheng!(new_A, copy_V, new_cache) |> wait
    # @time PartitionedArrays.psparse_yung_sheng!(new_A, copy_V, new_cache) |> wait
    # A,cache = psparse(I,J,V,row_partition,col_partition,split_format=false,reuse=true) |> fetch
    # psparse!(A,V,cache) |> wait
end

PartitionedArrays.with_debug(main)
