using MPI
using CUDA

MPI.Init()
comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)
dst = mod(rank+1, size)
src = mod(rank-1, size)
println("rank=$rank, size=$size, dst=$dst, src=$src")
N = 2
send_mesg = CuArray{Float64}(undef, N)
recv_mesg = CuArray{Float64}(undef, N)

CUDA.fill!(send_mesg, Float64(rank))
CUDA.fill!(recv_mesg, Float64(0))

# println(Array(send_mesg))
#rreq = MPI.Irecv!(recv_mesg, src,  src+32, comm)
MPI.Sendrecv!(send_mesg, dst, 0, recv_mesg, src, 0, comm)

# println("recv_mesg on proc $rank: $recv_mesg")