crate::ix!();

pub struct BlobState {
    total_size:      i64,
    current_size:    i64,
    is_tensor:       bool,
    seen_chunks_ids: HashSet<i32>,
}

impl BlobState {
    
    pub fn new(total_size: i64, current_size: i64, is_tensor: bool) -> Self {
        todo!();
        /*
            : total_size(total_size),
            current_size(current_size),
            is_tensor(is_tensor)
        */
    }
}
