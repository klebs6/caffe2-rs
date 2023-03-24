crate::ix!();

pub enum ParallelizationScheme {
    none,
    split_by_batch,
    split_by_length,
    shard,
    shard_by_number
}
