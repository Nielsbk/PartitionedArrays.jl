using MPI
using CUDA

function main()
    MPI.Init()
    comm = MPI.COMM_WORLD
    rank = MPI.Comm_rank(comm)
    size = MPI.Comm_size(comm)

    N = 10
    gpu_buf = CUDA.fill(rank, N)

    if rank == 0
        all_data = CUDA.zeros(Int, N * size)
        all_data[1:N] .= gpu_buf

        for src in 1:size-1
            offset = src * N
            MPI.Irecv!(view(all_data, offset+1:offset+N), src, 0, comm)
        end

        println("Rank 0 received:")
        host_data = Array(reshape(all_data, N, size))
        println(host_data)

    else
        MPI.Isend(gpu_buf, 0, 0, comm)
    end

    MPI.Barrier(comm)
    MPI.Finalize()
end

main()