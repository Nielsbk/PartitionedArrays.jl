# examples/04-sendrecv.jl
using MPI
using CUDA
using Adapt

MPI.Init()

comm = MPI.COMM_WORLD
rank = MPI.Comm_rank(comm)
size = MPI.Comm_size(comm)

dst = mod(rank+1, size)
src = mod(rank-1, size)

N = 4

send_mesg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
recv_mesg = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

fill!(send_mesg, Float64(rank))

send_mesg = CuArray(send_mesg)
recv_mesg = CuArray(recv_mesg)

rreq = MPI.Irecv!(recv_mesg, comm; source=src, tag=src+32)

print("$rank: Sending   $rank -> $dst = $Array(send_mesg)\n")
sreq = MPI.Isend(send_mesg, comm; dest=dst, tag=rank+32)

stats = MPI.Waitall([rreq, sreq])

print("$rank: Received $src -> $rank = $Array(recv_mesg)\n")

MPI.Barrier(comm)