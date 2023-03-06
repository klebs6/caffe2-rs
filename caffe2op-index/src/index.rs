crate::ix!();

pub struct Index<T> {
    base: IndexBase,
    dict: HashMap<T,i64>,
}

impl<T> Index<T> {
    
    pub fn new(max_elements: i64) -> Self {
        todo!();
        /*
            : IndexBase(maxElements, TypeMeta::Make<T>())
        */
    }
    
    #[inline] pub fn get(
        &mut self, 
        keys:     *const T,
        values:   *mut i64,
        num_keys: usize)  
    {
        
        todo!();
        /*
            if (frozen_) {
          FrozenGet(keys, values, numKeys);
          return;
        }
        std::lock_guard<std::mutex> lock(dictMutex_);
        for (int i = 0; i < numKeys; ++i) {
          auto it = dict_.find(keys[i]);
          if (it != dict_.end()) {
            values[i] = it->second;
          } else if (nextId_ < maxElements_) {
            auto newValue = nextId_++;
            dict_.insert({keys[i], newValue});
            values[i] = newValue;
          } else {
            CAFFE_THROW("Dict max size reached");
          }
        }
        */
    }
    
    #[inline] pub fn load(
        &mut self, 
        keys: *const T,
        num_keys: usize) -> bool 
    {
        todo!();
        /*
            CAFFE_ENFORCE(
            numKeys <= maxElements_,
            "Cannot load index: Tensor is larger than max_elements.");
        decltype(dict_) dict;
        for (auto i = 0U; i < numKeys; ++i) {
          CAFFE_ENFORCE(
              dict.insert({keys[i], i + 1}).second,
              "Repeated elements found: cannot load into dictionary.");
        }
        // assume no `get` is inflight while this happens
        {
          std::lock_guard<std::mutex> lock(dictMutex_);
          // let the old dict get destructed outside of the lock
          dict_.swap(dict);
          nextId_ = numKeys + 1;
        }
        return true;
        */
    }
    
    #[inline] pub fn store(&mut self, out: *mut Tensor) -> bool {
        
        todo!();
        /*
            std::lock_guard<std::mutex> lock(dictMutex_);
        out->Resize(nextId_ - 1);
        auto outData = out->template mutable_data<T>();
        for (const auto& entry : dict_) {
          outData[entry.second - 1] = entry.first;
        }
        return true;
        */
    }
    
    #[inline] pub fn frozen_get(
        &mut self, 
        keys: *const T,
        values: *mut i64,
        num_keys: usize)
    {
        todo!();
        /*
            for (auto i = 0U; i < numKeys; ++i) {
          auto it = dict_.find(keys[i]);
          values[i] = it != dict_.end() ? it->second : 0;
        }
        */
    }
}
