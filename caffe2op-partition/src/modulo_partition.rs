crate::ix!();

#[inline] pub fn modulo_partition<Index>(key: Index, num_partitions: i32) -> i32 {
    todo!();
    /*
        int shard = key % numPartitions;
      // equivalent to `if (shard < 0) shard += partitions;`
      shard += numPartitions & (shard >> (sizeof(int) * 8 - 1));
      return shard;
    */
}
