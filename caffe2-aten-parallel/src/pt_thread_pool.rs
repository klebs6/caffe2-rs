crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/PTThreadPool.h]

pub struct PTThreadPool {
    base: ThreadPool,
}

impl PTThreadPool {

    pub fn new(
        pool_size:    i32,
        numa_node_id: i32) -> Self {

        let numa_node_id: i32 = numa_node_id.unwrap_or(-1);

        todo!();

        /*


            : ThreadPool(pool_size, numa_node_id, [](){
            setThreadName("PTThreadPool");
            init_num_threads();
          })
        */
    }
}
