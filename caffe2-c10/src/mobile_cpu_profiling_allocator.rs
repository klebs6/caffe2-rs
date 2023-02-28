crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/mobile/CPUProfilingAllocator.h]

/**
  | Given a sequence of allocations in a
  | thread, AllocationPlan records
  | 
  | - 1. size of each allocation
  | 
  | - 2. Lifetime of each allocation.
  | 
  | - 3. allocation offsets: Memory offset
  | for each allocation in a single blob
  | of memory
  | 
  | - 4. Total size of a blob of memory required
  | to satisfy all the allocations.
  |
  */
pub struct AllocationPlan {

    /**
     | Records size of each allocation by their
     | sequential allocation ids.
     |
     */
    allocation_sizes:     Vec<u64>,

    /**
     | This maps one allocation id (X) to another
     | allocation id (Y).
     | 
     | Allocation X is alive until allocation
     | Y. From allocation Y onwards allocation
     | X is not referenced.
     | 
     | Thus Y is the id of the first allocation
     | after X is freed.
     | 
     | NB: When an allocation is recorded,
     | along with recording its size, we also
     | set the lifetime to be numeric_limits::max()
     | 
     | This is to track allocations that are
     | made during the scope of profiling but
     | were not freed until after the scope
     | ended.
     | 
     | Such allocations are not managed by
     | profiling allocator.
     |
     */
    allocation_lifetimes: Vec<u64>,

    /**
      | Maps an allocation to some offset in
      | a blob of memory.
      |
      */
    allocation_offsets:   Vec<u64>,

    total_size:           u64, // default = { 0 }
}

/**
  | Map of memory ptr to allocation id. This
  | is auxiliary information only used
  | to establish lifetime of allocations.
  |
  */
pub struct AllocationPlanner {

    allocation_plan:      *mut AllocationPlan, // default = { nullptr }

    /**
      | Maps allocated ptr to its allocation
      | id.
      | 
      | This is used when freeing the memory
      | to lookup the allocation id in order
      | to establish the lifetime of a particular
      | allocation.
      |
      */
    allocation_ptr_to_id: HashMap<*const c_void,u64>, // was FlatHashMap

    allocation_id:      u64, // default = { 0 }
    validation_mode:    bool, // default = { false }
    validation_success: bool, // default = { true }
}

impl AllocationPlanner {
    
    pub fn new(
        plan:     *mut AllocationPlan,
        validate: Option<bool>) -> Self {

        let validate: bool = validate.unwrap_or(false);

        todo!();
        /*
        : allocation_plan(plan),
        : validation_mode(validate),

        
        */
    }
}

// NOT THREAD SAFE profiling allocator.
pub struct CPUProfilingAllocator {
    plan:                 *const AllocationPlan, // default = { nullptr }
    allocation_id:        u64, // default = { 0 }
    current_size:         u64, // default = { 0 }
    blob:                 *mut c_void, // default = { nullptr }
    allocation_ptr_to_id: HashMap<*const c_void,u64>, // was FlatHashMap
}

/**
 | Usage: Profile allocations made by one run of
 | the model.
 |
 | AllocationPlan plan;
 |
 | {
 |   WithProfileAllocationGuard profile_guard(&plan);
 |   module.forward(...);
 | }
 | plan now contains allocation plan.
 */
pub struct WithProfileAllocationsGuard {
    planner: Box<AllocationPlanner>,
}

/*
 | Usage: Validate allocation plan made with
 | WithProfileAllocationGuard
 |
 | bool plan_validation_success, success = true;
 |
 | for (some number of representative inputs)
 | {
 |   WithValidateAllocationPlanGuard(&plan, &plan_validation_success);
 |   module.forward(...);
 |   success = success && plan_validation_success;
 | }
 | success == true means allocations are according to plan
 | else for some inputs allocation pattern changed.
 */
pub struct WithValidateAllocationPlanGuard {
    planner: Box<AllocationPlanner>,
    success: *mut bool,
}

/*
 | Usage: Allocate tensors accordingly to allocation plan
 | First make allocation plan.
 |  See WithProfileAllocationsGuard usage.
 | Second validate allocation plan.
 |  See WithValidateAllocationPlanGuard usage.
 | CPUProfilingAllocator profiling_allocator;
 | {
 |   WithProfilingAllocatorGuard allocator_guard(&profiling_allocator, &plan);
 |   module.forward(...);
 | }
 */
pub struct WithProfilingAllocatorGuard {

}

//-------------------------------------------[.cpp/pytorch/c10/mobile/CPUProfilingAllocator.cpp]

lazy_static!{
    /*
    thread_local AllocationPlanner* allocation_planner{nullptr};
    thread_local CPUProfilingAllocator* profiling_allocator{nullptr};
    */
}

#[derive(PartialEq,Eq)]
pub struct MemBlock {
    start_offset: u64,
    end_offset:   u64,
}

impl MemBlock {
    
    pub fn new(s: u64, e: u64) -> Self {
    
        todo!();
        /*
        : start_offset(s),
        : end_offset(e),

        
        */
    }
}

impl Ord for MemBlock {

    #[inline] fn cmp(&self, other: &MemBlock) -> Ordering {
        todo!();
        /*
            return start_offset < other.start_offset;
        */
    }
}

impl PartialOrd<MemBlock> for MemBlock {
    
    #[inline] fn partial_cmp(&self, other: &MemBlock) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

pub enum EventType { 
    Allocate = 0, 
    Free, 
    Invalid 
}

pub struct MemEvent {
    time:          u64,
    allocation_id: u64,
    size:          u64,

    /**
      | {EventType::Invalid};
      |
      */
    ty:            EventType,
}

impl MemEvent {
    
    pub fn new(
        t:  u64,
        id: u64,
        s:  u64,
        e:  EventType) -> Self {
    
        todo!();
        /*
        : time(t),
        : allocation_id(id),
        : size(s),
        : ty(e),

        
        */
    }
}

pub fn overlaps(
    a: &MemBlock,
    b: &MemBlock) -> bool {
    
    todo!();
        /*
            // two blocks dont overlap if
      // |---a--------|--------------b--------|
      // strat_a     end_a <= start_b       end_b
      return !(
          (a.end_offset <= b.start_offset) || (b.end_offset <= a.start_offset));
        */
}

pub fn validate_allocation_plan(
    alloc_events:       &Vec<MemEvent>,
    allocation_offsets: &Vec<u64>) -> bool {
    
    todo!();
        /*
            set<MemBlock> allocations;
      for (const auto& event : alloc_events) {
        auto alloc_id = event.allocation_id;
        // Skip allocations not managed by AllocationPlan
        if (allocation_offsets[alloc_id] == uint64_t::max) {
          continue;
        }
        auto start_offset = allocation_offsets[alloc_id];
        auto end_offset = allocation_offsets[alloc_id] + event.size;
        MemBlock mem_block(start_offset, end_offset);
        if (event.type == EventType::Allocate) {
          auto it = allocations.lower_bound(mem_block);
          if (it != allocations.end()) {
            auto next_block = *it;
            if (overlaps(next_block, mem_block)) {
              return false;
            }
          }
          if (it != allocations.begin()) {
            auto prev_block = *(--it);
            if (overlaps(prev_block, mem_block)) {
              return false;
            }
          }
          allocations.emplace(mem_block);
        } else if (event.type == EventType::Free) {
          auto it = allocations.find(mem_block);
          TORCH_CHECK(
              (*it).end_offset == end_offset,
              "Enf offset of allocation being freed must match the one recorded.");
          TORCH_CHECK(
              it != allocations.end(),
              "ProfilingAllocator: Allocate event "
              "must have preceded deallocate event.");
          allocations.erase(it);
        } else {
          TORCH_CHECK(false, "ProfilingAllocator: Invalid event type.");
        }
      }
      return true;
        */
}

pub fn create_and_sort_mem_events(
    allocation_sizes:     &Vec<u64>,
    allocation_lifetimes: &Vec<u64>) -> Vec<MemEvent> {
    
    todo!();
        /*
            vector<MemEvent> events;
      for (uint64_t i = 0; i < allocation_sizes.size(); ++i) {
        // If observed allocation are freed outside the scope of
        // observation, then allocations are not managed by the
        // AllocationPlan.
        if (allocation_lifetimes[i] == uint64_t::max) {
          continue;
        }
        events.emplace_back(i, i, allocation_sizes[i], EventType::Allocate);
        events.emplace_back(
            allocation_lifetimes[i], i, allocation_sizes[i], EventType::Free);
      }
      sort(
          events.begin(),
          events.end(),
          [](const MemEvent& a, const MemEvent& b) -> bool {
            return a.time < b.time;
          });
      return events;
        */
}

pub fn formulate_greedy_allocation_plan(
    allocation_sizes:     &Vec<u64>,
    allocation_lifetimes: &Vec<u64>) -> Vec<u64> {
    
    todo!();
        /*
            // Step 1. Construct all allocation/free events.
      //         Sort these events by timestamp.
      // Step 2. Iterate through all events.
      //  2.1 If allocate event:
      //      Find all candidate in free_size_to_offset map
      //      Greedily pick the first one.
      //      Remove the entry from free_size_to_offset map.
      //      new_offset = offset + request_size
      //      new_size = size - request_size
      //      Add new entry to both maps
      //  2.2 If free event.
      //      Check if the returned offset merges with another chunk.
      //      If so merge until no more merging is possible.
      //      If returned offset does not merge, then
      //      just return it as a chunk.

      // lower_bound on this map will get all candidates of
      // the right size for allocation.
      map<uint64_t, uint64_t> free_size_to_offset;
      // This provides fast lookup when we want to insert freed block
      // back, especially when we want to merge blocks.
      ska::flat_hash_map<uint64_t, map<uint64_t, uint64_t>::iterator>
          free_start_offset_to_size_iter;
      ska::flat_hash_map<uint64_t, map<uint64_t, uint64_t>::iterator>
          free_end_offset_to_size_iter;
      // Upon free end_ptr = offset + size
      // If end_ptr exists merge freed allocation
      // Also find corresponding offset in size_to_offset
      // Remove that entry and update with new size and offset
      // If end_ptr does not exist then just insert offset,size
      // in map and correspondingly size, offset in the other map.
      // Merging should always be done recursively until no more chunks
      // that can be found.
      // After last free we should have only one entry left in these maps.

      vector<uint64_t> allocation_offsets(
          allocation_sizes.size(), uint64_t::max);
      auto mem_events =
          create_and_sort_mem_events(allocation_sizes, allocation_lifetimes);
      uint64_t max_offset{0};
      for (const auto& mem_event : mem_events) {
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        uint64_t alloc_offset;
        // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
        uint64_t new_offset, new_size;
        if (mem_event.type == EventType::Allocate) {
          auto it = free_size_to_offset.lower_bound(mem_event.size);
          if (it == free_size_to_offset.end()) {
            // If there is no contiguous block of the size requested
            // allocate a new one.
            alloc_offset = max_offset;
            max_offset += mem_event.size;
          } else {
            // If we have found a block of the size we want
            // 1. change the block by allocating out of it.
            //    1.1 Erase the entire block
            //    1.2 Erase the reverse map entries
            // 2. If block still has space left insert the remainder back in map.
            //    Including reverse map entries.
            alloc_offset = it->second;
            new_offset = alloc_offset + mem_event.size;
            new_size = it->first - mem_event.size;
            free_size_to_offset.erase(it);
            free_start_offset_to_size_iter.erase(alloc_offset);
            free_end_offset_to_size_iter.erase(alloc_offset + it->first);
            if (new_size > 0) {
              auto ref_it = free_size_to_offset.emplace(new_size, new_offset).first;
              free_start_offset_to_size_iter.emplace(new_offset, ref_it);
              free_end_offset_to_size_iter.emplace(new_offset + new_size, ref_it);
            }
          }
          allocation_offsets[mem_event.allocation_id] = alloc_offset;
        } else {
          // 1. Check if freed block is adjacent to an existing free block
          //    at its end boundary. This is done by checking
          //    free_end_offset_to_size_iter.
          //    If we find such a block, remove it and adjust size of
          //    the block being freed.
          // 2. Similarly check if freed block is adjacent to an existing
          //    free block at start boundary. This is done by checking
          //    free_start_offset_to_size_iter.
          //    If we find such a block, remove it and adjust size of
          //    the block being freed.
          // 3. Insert the freed block in map.
          auto freed_offset = allocation_offsets[mem_event.allocation_id];
          auto freed_size = mem_event.size;
          auto end_offset = freed_offset + freed_size;
          // Merge when another free block exist at the end of this block
          auto end_it = free_start_offset_to_size_iter.find(end_offset);
          if (end_it != free_start_offset_to_size_iter.end()) {
            auto merge_block_iter = end_it->second;
            auto merge_block_size = merge_block_iter->first;
            freed_size += merge_block_size;
            free_size_to_offset.erase(merge_block_iter);
            free_start_offset_to_size_iter.erase(end_it);
            // If the block is being merged then also remove it from
            // free_end_offset_to_size_iter
            free_end_offset_to_size_iter.erase(end_offset + merge_block_size);
          }
          // Merge when freed block exist at the end of another free block
          auto start_it = free_end_offset_to_size_iter.find(freed_offset);
          if (start_it != free_end_offset_to_size_iter.end()) {
            auto merge_block_iter = start_it->second;
            auto merge_block_size = merge_block_iter->first;
            freed_size += merge_block_size;
            freed_offset -= merge_block_size;
            free_size_to_offset.erase(merge_block_iter);
            free_end_offset_to_size_iter.erase(start_it);
            // If the block is being merged then also remove it from
            // free_start_offset_to_size_iter
            free_start_offset_to_size_iter.erase(freed_offset);
          }
          auto freed_block_it =
              free_size_to_offset.emplace(freed_size, freed_offset).first;
          free_start_offset_to_size_iter.emplace(freed_offset, freed_block_it);
          free_end_offset_to_size_iter.emplace(
              freed_offset + freed_size, freed_block_it);
        }
      }
      TORCH_CHECK(
          validate_allocation_plan(mem_events, allocation_offsets),
          "ProfilingAllocator: Allocation plan invalid.");
      return allocation_offsets;
        */
}

impl AllocationPlan {
    
    pub fn clear(&mut self)  {
        
        todo!();
        /*
            allocation_sizes.clear();
      allocation_lifetimes.clear();
      allocation_offsets.clear();
        */
    }
}

impl AllocationPlanner {
    
    pub fn record_allocation(&mut self, 
        size: u64,
        ptr:  *const c_void)  {
        
        todo!();
        /*
            if (validation_mode_) {
        validation_success = validation_success && validate_allocation(size, ptr);
        return;
      }
      allocation_plan_->allocation_sizes.push_back(size);
      allocation_plan_->allocation_lifetimes.push_back(
          uint64_t::max);
      allocation_ptr_to_id_[ptr] = allocation_id_;
      allocation_id_++;
        */
    }
    
    pub fn record_free(&mut self, ptr: *const c_void)  {
        
        todo!();
        /*
            if (validation_mode_) {
        validation_success = validation_success && validate_free(ptr);
        return;
      }
      auto it = allocation_ptr_to_id_.find(ptr);
      if (it == allocation_ptr_to_id_.end()) {
        // Free being recorded was allocated outside of WithProfileAllocationGuard
        return;
      }
      auto id = it->second;
      TORCH_CHECK(
          id < allocation_plan_->allocation_lifetimes.size(),
          "Allocation must have been recorded during record_allocation.");
      allocation_plan_->allocation_lifetimes[id] = allocation_id_;
        */
    }
    
    pub fn validate_allocation(&mut self, 
        size: u64,
        ptr:  *const c_void) -> bool {
        
        todo!();
        /*
            if (allocation_id_ >= allocation_plan_->allocation_sizes.size() ||
          allocation_plan_->allocation_sizes[allocation_id_] != size) {
        TORCH_WARN(
            "Allocation request does not match plan:",
            "Allocation id:",
            allocation_id_,
            ", Number of recorded allocations:",
            allocation_plan_->allocation_sizes.size(),
            ", Recorded size of the requested allocation:",
            allocation_plan_->allocation_sizes[allocation_id_],
            ", but got:",
            size);

        return false;
      }
      allocation_ptr_to_id_[ptr] = allocation_id_;
      allocation_id_++;
      return true;
        */
    }
    
    pub fn validate_free(&mut self, ptr: *const c_void) -> bool {
        
        todo!();
        /*
            auto it = allocation_ptr_to_id_.find(ptr);
      if (it == allocation_ptr_to_id_.end()) {
        // Allocation that was made outside the validation scope is being freed here
        return true;
      }
      auto id = (*it).second;
      TORCH_CHECK(
          id < allocation_plan_->allocation_lifetimes.size(),
          "Allocation must have been recorded during validate_allocation.");
      auto lifetime_id = allocation_plan_->allocation_lifetimes[id];
      return (lifetime_id == allocation_id_);
        */
    }
    
    pub fn formulate_plan(&mut self)  {
        
        todo!();
        /*
            allocation_plan_->allocation_offsets = formulate_greedy_allocation_plan(
          allocation_plan_->allocation_sizes,
          allocation_plan_->allocation_lifetimes);
      allocation_plan_->total_size = 0;
      for (const auto i : irange(allocation_plan_->allocation_sizes.size())) {
        if (allocation_plan_->allocation_lifetimes[i] ==
            uint64_t::max) {
          continue;
        }
        auto limit = allocation_plan_->allocation_offsets[i] +
            allocation_plan_->allocation_sizes[i];
        allocation_plan_->total_size =
            max(allocation_plan_->total_size, limit);
      }
        */
    }
    
    pub fn clear(&mut self)  {
        
        todo!();
        /*
            allocation_plan_->clear();
      allocation_ptr_to_id_.clear();
        */
    }
}

impl CPUProfilingAllocator {
    
    pub fn set_plan(&mut self, plan: *const AllocationPlan)  {
        
        todo!();
        /*
            TORCH_CHECK(plan != nullptr, "Allocation plan is nullptr.");
      plan_ = plan;
      allocation_id_ = 0;
      allocation_ptr_to_id_.clear();
      if (current_size_ < plan->total_size) {
        // Free existing memory and reallocate for larger size.
        free_cpu(blob_);
        blob_ = alloc_cpu(plan->total_size);
        current_size_ = plan->total_size;
      }
        */
    }
    
    pub fn unset_plan(&mut self)  {
        
        todo!();
        /*
            allocation_id_ = 0;
      allocation_ptr_to_id_.clear();
      plan_ = nullptr;
        */
    }
    
    pub fn allocate(&mut self, bytes: usize)  {
        
        todo!();
        /*
            TORCH_CHECK(
          bytes == plan_->allocation_sizes[allocation_id_],
          "Got allocation request that does not match with the plan.");
      if (plan_->allocation_lifetimes[allocation_id_] ==
          uint64_t::max) {
        // This allocation is not managed by ProfilingAllocator.
        allocation_id_++;
        return alloc_cpu(bytes);
      }
      void* ptr = reinterpret_cast<uint8_t*>(blob_) +
          plan_->allocation_offsets[allocation_id_];
      allocation_ptr_to_id_[ptr] = allocation_id_;
      allocation_id_++;
      return ptr;
        */
    }
    
    pub fn free(&mut self, ptr: *mut c_void)  {
        
        todo!();
        /*
            auto it = allocation_ptr_to_id_.find(ptr);
      if (it == allocation_ptr_to_id_.end()) {
        // Either
        // 1. Allocation that was made outside the validation scope is being freed
        // here or
        // 2. Allocation that is not managed by profiling allocator is being freed.
        //    Example of the second type
        //    Tensor out;
        //    for (....) {
        //      {
        //        CPUProfilingAllocator
        //        out = ...some op (This also frees previous memory held by out)
        //      }
        //      out is used..
        //    }
        free_cpu(ptr);
        return;
      }
      auto id = it->second;
      TORCH_CHECK(
          id < plan_->allocation_lifetimes.size(),
          "Freeing allocation that is not accordingly to the plan.");
      auto lifetime_id = plan_->allocation_lifetimes[id];
      TORCH_CHECK(
          lifetime_id == allocation_id_,
          "Lifetime of allocations do not match: allocation_id ",
          id,
          ", expected:",
          lifetime_id,
          ", got:",
          allocation_id_);
        */
    }
}

impl Drop for CPUProfilingAllocator {

    fn drop(&mut self) {
        todo!();
        /*
            free_cpu(blob_);
        */
    }
}

impl WithProfileAllocationsGuard {

    pub fn new(plan: *mut AllocationPlan) -> Self {
    
        todo!();
        /*


            // Nesting of allocation profiling does not seem meaningful.
      TORCH_CHECK(
          allocation_planner == nullptr,
          "Nesting profiling allocations is not supported.");
      planner_ = make_unique<AllocationPlanner>(plan);
      planner_->clear();
      allocation_planner = planner_.get();
        */
    }
}

impl Drop for WithProfileAllocationsGuard {

    fn drop(&mut self) {
        todo!();
        /*
            planner_->formulate_plan();
      allocation_planner = nullptr;
        */
    }
}

impl WithValidateAllocationPlanGuard {
    
    pub fn new(
        plan:    *mut AllocationPlan,
        success: *mut bool) -> Self {
    
        todo!();
        /*


            // Nesting of allocation profiling does not seem meaningful.
      TORCH_CHECK(
          allocation_planner == nullptr,
          "Nesting profiling allocations is not supported.");
      planner_ = make_unique<AllocationPlanner>(plan, true);
      success_ = success;
      allocation_planner = planner_.get();
        */
    }
}

impl Drop for WithValidateAllocationPlanGuard {

    fn drop(&mut self) {
        todo!();
        /*
            *success_ = planner_->validation_success;
      allocation_planner = nullptr;
        */
    }
}

pub fn get_thread_local_allocation_planner() -> *mut AllocationPlanner {
    
    todo!();
        /*
            return allocation_planner;
        */
}

impl WithProfilingAllocatorGuard {
    
    pub fn new(
        allocator: *mut CPUProfilingAllocator,
        plan:      *const AllocationPlan) -> Self {
    
        todo!();
        /*


            // Nesting of profiling allocator is not supported.
      TORCH_CHECK(
          profiling_allocator == nullptr,
          "Nesting profiling allocators is not supported.");
      profiling_allocator = allocator;
      profiling_allocator->set_plan(plan);
        */
    }
}

impl Drop for WithProfilingAllocatorGuard {

    fn drop(&mut self) {
        todo!();
        /*
            profiling_allocator->unset_plan();
      profiling_allocator = nullptr;
        */
    }
}

pub fn get_thread_local_profiling_allocator() -> *mut CPUProfilingAllocator {
    
    todo!();
        /*
            return profiling_allocator;
        */
}
