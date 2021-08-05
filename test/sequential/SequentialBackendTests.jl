module SequentialBackendTests

using PartitionedArrays
using Test

function main(parts)

  nparts = num_parts(parts)
  @assert nparts == 4

  @test i_am_main(parts) == true
  
  values = map_parts(parts) do part
    10*part
  end

  @test size(values) == size(parts)
  
  map_parts(parts,values) do part, value
    @test 10*part == value
  end
  
  parts_rcv = map_parts(parts) do part
    if part == 1
      [2,3]
    elseif part == 2
      [4,]
    elseif part == 3
      [1,2]
    else
      [1,3]
    end
  end
  
  parts_snd = map_parts(parts) do part
    if part == 1
      [3,4]
    elseif part == 2
      [1,3]
    elseif part == 3
      [1,4]
    else
      [2]
    end
  end
  
  data_snd = map_parts(i->10*i,parts_snd)
  data_rcv = map_parts(similar,parts_rcv)
  
  t = async_exchange!(
    data_rcv,
    data_snd,
    parts_rcv,
    parts_snd)
  
  map_parts(i->isa(i,Task),t)
  map_parts(schedule,t)
  map_parts(wait,t)
  
  map_parts(parts,data_rcv) do part, data_rcv
    if part == 1
      r = [10,10]
    elseif part == 2
      r = [20]
    elseif part == 3
      r = [30,30]
    else
      r= [40,40]
    end
    @test r == data_rcv
  end
  
  data_snd = map_parts(parts,parts_snd) do part, parts_snd
    Table([ Int[i,part] for i in parts_snd])
  end

  @test size(data_snd) == size(parts)
  
  data_rcv = exchange(data_snd,parts_rcv,parts_snd)
  
  map_parts(parts,data_rcv) do part, data_rcv
    if part == 1
      r = [[1,2],[1,3]]
    elseif part == 2
      r = [[2,4]]
    elseif part == 3
      r = [[3,1],[3,2]]
    else
      r= [[4,1],[4,3]]
    end
    @test Table(r) == data_rcv
  end
  
  a_and_b = map_parts(parts) do part
    10*part, part+10
  end
  
  a,b = a_and_b
  map_parts(a_and_b,a,b) do a_and_b, a, b
    a1,b1 = a_and_b
    @test a == a1
    @test b == b1
  end

  snd = map_parts(parts) do part
    if part == 1
      [1,2]
    elseif part == 2
      [2,3,4]
    elseif part == 3
      [5,6]
    else
      [7,8,9,10]
    end
  end
  rcv = gather(snd) 
  map_parts(parts,rcv) do part, rcv
    if part == MAIN
      @test rcv == [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    else
      @test rcv == Vector{Int}[]
    end
    @test isa(rcv,Table)
  end

  rcv = gather_all(snd) 
  map_parts(rcv) do rcv
    @test rcv == [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    @test isa(rcv,Table)
  end
  
  rcv = gather(parts) 
  @test size(rcv) == size(parts)
  
  map_parts(parts,rcv) do part, rcv
    if part == MAIN
      @test rcv == collect(1:nparts)
    else
      @test length(rcv) == 0
    end
  end
  
  @test get_main_part(rcv) == collect(1:nparts)
  
  rcv = scatter(rcv)
  map_parts(parts,rcv) do part, rcv
    @test part == rcv
  end

  snd = map_parts(parts) do part
    if part == MAIN
      v = [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    else
      v = Vector{Int}[]
    end
    Table(v)
  end
  rcv = scatter(snd)
  map_parts(parts,rcv) do part,rcv
    if part == 1
      r = [1,2]
    elseif part == 2
      r = [2,3,4]
    elseif part == 3
      r = [5,6]
    else
      r= [7,8,9,10]
    end
    @test r == rcv
  end

  snd = map_parts(parts) do part
    if part == MAIN
      v = [[1,2],[2,3,4],[5,6],[7,8,9,10]]
    else
      v = Vector{Int}[]
    end
    v
  end
  rcv = scatter(snd)
  map_parts(parts,rcv) do part,rcv
    if part == 1
      r = [1,2]
    elseif part == 2
      r = [2,3,4]
    elseif part == 3
      r = [5,6]
    else
      r= [7,8,9,10]
    end
    @test r == rcv
  end

  rcv = gather_all(parts) 
  
  map_parts(rcv) do rcv
    @test rcv == collect(1:nparts)
  end
  
  @test get_part(rcv) == collect(1:nparts)
  
  rcv = emit(parts)

  @test size(rcv) == size(parts)
  
  map_parts(rcv) do rcv
    @test rcv == 1
  end

end

#nparts = 4
#main(get_part_ids(sequential,nparts))
#
#nparts = (2,2)
#main(get_part_ids(sequential,nparts))

function main_ml(h)

  l1_parts =h.curr[1]
  l2_to_l1 = h.prev[2]
  l1_to_l2 = h.next[1]
  
  l1_data = map_parts(i->10*i,l1_parts)
  l2_data = gather_next(l1_data,l2_to_l1)
  map_parts(l2_data,l2_to_l1) do l2,l1
    @test l2 == 10*l1
  end

  l1_data = map_parts(i->i*collect(1:4*(mod(i,3)+1)),l1_parts)
  l2_data = gather_next(l1_data,l2_to_l1)
  map_parts(l2_data,l2_to_l1) do l2,l1
    @test l2 == map(i->i*collect(1:4*(mod(i,3)+1)),l1)
  end
end

#h = Hierarchy(sequential,[(8,8),(2,2),(1,1)])
#main_ml(h)

h = Hierarchy(sequential,[10,3,1])
main_ml(h)




end # module
