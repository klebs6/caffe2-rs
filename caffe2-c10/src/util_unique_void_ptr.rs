crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/util/UniqueVoidPtr.h]

/**
  | dyn std::alloc::Allocator UniqueVoidPtr is an owning smart pointer like
  | unique_ptr, but with three major differences:
  |
  |    1) It is specialized to void
  |
  |    2) It is specialized for a function pointer
  |       deleter void(void* ctx); i.e., the
  |       deleter doesn't take a reference to the
  |       data, just to a context pointer (erased
  |       as void*).  In fact, internally, this
  |       pointer is implemented as having an
  |       owning reference to context, and
  |       a non-owning reference to data; this is
  |       why you release_context(), not release()
  |       (the conventional API for release()
  |       wouldn't give you enough information to
  |       properly dispose of the object later.)
  |
  |    3) The deleter is guaranteed to be called
  |       when the unique pointer is destructed and
  |       the context is non-null; this is
  |       different from unique_ptr where the
  |       deleter is not called if the data pointer
  |       is null.
  |
  | Some of the methods have slightly different
  | types than unique_ptr to reflect this.
  |
  */
pub struct UniqueVoidPtr {

    /**
      | Lifetime tied to ctx_
      |
      */
    data: *mut c_void,
    ctx:  Box<*mut c_void,&'static dyn std::alloc::Allocator>,
}

impl Default for UniqueVoidPtr {
    
    fn default() -> Self {
        todo!();
        /*


            : data_(nullptr), ctx_(nullptr, &deleteNothing)
        */
    }
}

impl Deref for UniqueVoidPtr {

    type Target = c_void;
    
    #[inline] fn deref(&self) -> &Self::Target {

        todo!();
        /*
            return data_;
        */
    }
}

impl UniqueVoidPtr {

    pub fn new(data: *mut c_void) -> Self {
    
        todo!();
        /*
            : data_(data), ctx_(nullptr, &deleteNothing)
        */
    }
    
    pub fn new_with_deleter<A: std::alloc::Allocator>(
        data:        *mut c_void,
        ctx:         *mut c_void,
        ctx_deleter: A) -> Self {
    
        todo!();
        /*
            : data_(data), ctx_(ctx, ctx_deleter ? ctx_deleter : &deleteNothing)
        */
    }
    
    pub fn clear(&mut self)  {
        
        todo!();
        /*
            ctx_ = nullptr;
        data_ = nullptr;
        */
    }
    
    pub fn get(&self)  {
        
        todo!();
        /*
            return data_;
        */
    }
    
    pub fn get_context(&self)  {
        
        todo!();
        /*
            return ctx_.get();
        */
    }
    
    pub fn release_context(&mut self)  {
        
        todo!();
        /*
            return ctx_.release();
        */
    }
    
    pub fn move_context<A2: std::alloc::Allocator>(&mut self) -> &mut Box<*mut c_void,A2> {
        
        todo!();
        /*
            return move(ctx_);
        */
    }
    
    pub fn compare_exchange_deleter<A,B>(&mut self, 
        expected_deleter: A,
        new_deleter:      B) -> bool 
        where 
            A: std::alloc::Allocator,
            B: std::alloc::Allocator,
    {
        
        todo!();
        /*
            if (get_deleter() != expected_deleter)
          return false;
        ctx_ = unique_ptr<void, dyn std::alloc::Allocator>(ctx_.release(), new_deleter);
        return true;
        */
    }
    
    pub fn cast_context<T,A: std::alloc::Allocator>(&self, expected_deleter: A) -> *mut T {
    
        todo!();
        /*
            if (get_deleter() != expected_deleter)
          return nullptr;
        return static_cast<T*>(get_context());
        */
    }
    
    pub fn operator_bool(&self) -> bool {
        
        todo!();
        /*
            return data_ || ctx_;
        */
    }
    
    pub fn get_deleter<A: std::alloc::Allocator>(&self) -> A {
        
        todo!();
        /*
            return ctx_.get_deleter();
        */
    }
}

/**
  | Note [How UniqueVoidPtr is implemented]
  | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  | 
  | UniqueVoidPtr solves a common problem
  | for allocators of tensor data, which
  | is that the data pointer (e.g., float*)
  | which you are interested in, is not the
  | same as the context pointer (e.g., DLManagedTensor)
  | which you need to actually deallocate
  | the data. Under a conventional deleter
  | design, you have to store extra context
  | in the deleter itself so that you can
  | actually delete the right thing.
  | 
  | Implementing this with standard C++
  | is somewhat error-prone: if you use
  | a unique_ptr to manage tensors, the
  | deleter will not be called if the data
  | pointer is nullptr, which can cause
  | a leak if the context pointer is non-null
  | (and the deleter is responsible for
  | freeing both the data pointer and the
  | context pointer).
  | 
  | So, in our reimplementation of unique_ptr,
  | which just store the context directly
  | in the unique pointer, and attach the
  | deleter to the context pointer itself.
  | In simple cases, the context pointer
  | is just the pointer itself.
  |
  */
lazy_static!{
    /*
    inline bool operator==(const UniqueVoidPtr& sp, nullptr_t)  {
      return !sp;
    }
    inline bool operator==(nullptr_t, const UniqueVoidPtr& sp)  {
      return !sp;
    }
    inline bool operator!=(const UniqueVoidPtr& sp, nullptr_t)  {
      return sp;
    }
    inline bool operator!=(nullptr_t, const UniqueVoidPtr& sp)  {
      return sp;
    }
    */
}

//-------------------------------------------[.cpp/pytorch/c10/util/UniqueVoidPtr.cpp]
