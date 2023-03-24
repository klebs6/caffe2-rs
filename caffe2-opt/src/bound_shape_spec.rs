crate::ix!();

/**
  | This struct stores the max bound size for batch
  | in the general sense. max_batch_size is the
  | upper bound of batch_size.
  |
  | max_seq_size is the upper bound of length of
  | every item in a batch.
  |
  | Upper bound of length of a batch of items
  | should be max_batch_size * max_seq_size.
  */
pub struct BoundShapeSpec {

    max_batch_size:    i64,
    max_seq_size:      i64, /// The following two parameters are for shape inference of UnPackRecords
    num_embeddings:    i64,
    embedding_length:  i64,
}

impl BoundShapeSpec {
    
    pub fn new(max_batch_size: i64, max_seq_size: i64) -> Self {
    
        todo!();
        /*
            : max_batch_size(b),
            max_seq_size(q),
            num_embeddings(0),
            embedding_length(0)
        */
    }
    
    pub fn new_with_embeddings(
        max_batch_size: i64,
        max_seq_size: i64,
        num_embeddings: i64,
        embedding_length: i64) -> Self {

        todo!();
        /*
            : max_batch_size(b),
            max_seq_size(q),
            num_embeddings(n),
            embedding_length(e)
        */
    }
}

