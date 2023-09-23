crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/CPUGeneratorImpl.h]

pub struct CPUGeneratorImpl {
    base:                      GeneratorImpl,
    engine:                    Mt19937,
    next_float_normal_sample:  Option<f32>,
    next_double_normal_sample: Option<f64>,
}

impl CPUGeneratorImpl {
    
    pub fn new(seed_in: u64) -> Self {
        let seed_in: u64 = seed_in.unwrap_or(default_rng_seed_val);
        todo!();
        /*
        
        */
    }

    // CPUGeneratorImpl methods
    pub fn clone(&self) -> Arc<CPUGeneratorImpl> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_current_seed(&mut self, seed: u64)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn current_seed(&self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn seed(&mut self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_state(&mut self, new_state: &TensorImpl)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn get_state(&self) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn device_type() -> DeviceType {
        
        todo!();
        /*
        
        */
    }
    
    pub fn random(&mut self) -> u32 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn random64(&mut self) -> u64 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn next_float_normal_sample(&mut self) -> Option<f32> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn next_double_normal_sample(&mut self) -> Option<f64> {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_next_float_normal_sample(&mut self, randn: Option<f32>)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_next_double_normal_sample(&mut self, randn: Option<f64>)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn engine(&mut self) -> Mt19937 {
        
        todo!();
        /*
        
        */
    }
    
    pub fn set_engine(&mut self, engine: Mt19937)  {
        
        todo!();
        /*
        
        */
    }
    
    pub fn clone_impl(&self) -> *mut CPUGeneratorImpl {
        
        todo!();
        /*
        
        */
    }
}

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/CPUGeneratorImpl.cpp]

/**
  | CPUGeneratorImplStateLegacy is a
  | POD class needed for memcpys in torch.get_rng_state()
  | and torch.set_rng_state().
  | 
  | It is a legacy class and even though it
  | is replaced with at::CPUGeneratorImpl,
  | we need this class and some of its fields
  | to support backward compatibility
  | on loading checkpoints.
  |
  */
pub struct CPUGeneratorImplStateLegacy {

    /**
     | The initial seed.
     |
     */
    the_initial_seed: u64,

    /**
     | = 1;
     |
     */
    left:             i32,

    /**
     | = 0;
     |
     */
    seeded:           i32,

    next:             u64,


    /**
     | the array for the state vector
     |
     */
    state: [u64; MERSENNE_STATE_N],

    /********************************/

    /**
     | For normal distribution
     |
     */
    normal_x:        f64,
    normal_y:        f64,
    normal_rho:      f64,

    /**
     | = 0;
     |
     */
    normal_is_valid: i32,
}

/**
  | CPUGeneratorImplState is a POD class
  | containing new data introduced in at::CPUGeneratorImpl
  | and the legacy state.
  | 
  | It is used as a helper for torch.get_rng_state()
  | and torch.set_rng_state() functions.
  |
  */
pub struct CPUGeneratorImplState {
    legacy_pod:                        CPUGeneratorImplStateLegacy,
    next_float_normal_sample:          f32,
    is_next_float_normal_sample_valid: bool,
}

/**
  | PyTorch maintains a collection of default
  | generators that get initialized once.
  | 
  | The purpose of these default generators
  | is to maintain a global running state
  | of the pseudo random number generation,
  | when a user does not explicitly mention
  | any generator. getDefaultCPUGenerator
  | gets the default generator for a particular
  | device.
  |
  */
pub fn get_default_cpu_generator<'a>() -> &'a Generator {
    
    todo!();
        /*
            static auto default_gen_cpu = createCPUGenerator(c10::detail::getNonDeterministicRandom());
      return default_gen_cpu;
        */
}

/**
  | Utility to create a CPUGeneratorImpl.
  | Returns a shared_ptr
  |
  */
pub fn create_cpu_generator(seed_val: u64) -> Generator {

    let seed_val: u64 = seed_val.unwrap_or(default_rng_seed_val);
    
    todo!();
        /*
            return make_generator<CPUGeneratorImpl>(seed_val);
        */
}

/**
  | Helper function to concatenate two
  | 32 bit unsigned int and return them as
  | a 64 bit unsigned int
  |
  */
#[inline] pub fn make_64bits_from_32bits(hi: u32, lo: u32) -> u64 {
    
    todo!();
        /*
            return (static_cast<u64>(hi) << 32) | lo;
        */
}

impl CPUGeneratorImpl {
    
    /**
     | CPUGeneratorImpl class implementation
     |
     */
    pub fn new(seed_in: u64) -> Self {
    
        todo!();
        /*


            : c10::GeneratorImpl{Device(DeviceType::CPU), DispatchKeySet(c10::DispatchKey::CPU)},
        engine_{seed_in},
        next_float_normal_sample_{c10::optional<float>()},
        next_double_normal_sample_{c10::optional<double>()}
        */
    }

    /**
      | Manually seeds the engine with the seed
      | input
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn set_current_seed(&mut self, seed: u64)  {
        
        todo!();
        /*
            next_float_normal_sample_.reset();
      next_double_normal_sample_.reset();
      engine_ = mt19937(seed);
        */
    }

    /**
      | Gets the current seed of CPUGeneratorImpl.
      |
      */
    pub fn current_seed(&self) -> u64 {
        
        todo!();
        /*
            return engine_.seed();
        */
    }

    /**
      | Gets a nondeterministic random number
      | from /dev/urandom or time, seeds the
      | CPUGeneratorImpl with it and then returns
      | that number.
      | 
      | FIXME: You can move this function to
      | Generator.cpp if the algorithm in getNonDeterministicRandom
      | is unified for both CPU and CUDA
      |
      */
    pub fn seed(&mut self) -> u64 {
        
        todo!();
        /*
            auto random = c10::detail::getNonDeterministicRandom();
      this->set_current_seed(random);
      return random;
        */
    }

    /**
      | Sets the internal state of CPUGeneratorImpl.
      | The new internal state must be a strided
      | CPU byte tensor and of the same size as
      | either
      | 
      | CPUGeneratorImplStateLegacy (for
      | legacy CPU generator state) or
      | 
      | CPUGeneratorImplState (for new state).
      | 
      | FIXME: Remove support of the legacy
      | state in the future?
      |
      */
    pub fn set_state(&mut self, new_state: &TensorImpl)  {
        
        todo!();
        /*
            using detail::CPUGeneratorImplState;
      using detail::CPUGeneratorImplStateLegacy;

      static_assert(std::is_pod<CPUGeneratorImplStateLegacy>::value, "CPUGeneratorImplStateLegacy is not a PODType");
      static_assert(std::is_pod<CPUGeneratorImplState>::value, "CPUGeneratorImplState is not a PODType");

      static const usize size_legacy = sizeof(CPUGeneratorImplStateLegacy);
      static const usize size_current = sizeof(CPUGeneratorImplState);
      static_assert(size_legacy != size_current, "CPUGeneratorImplStateLegacy and CPUGeneratorImplState can't be of the same size");

      detail::check_rng_state(new_state);

      at::mt19937 engine;
      auto float_normal_sample = c10::optional<float>();
      auto double_normal_sample = c10::optional<double>();

      // Construct the state of at::CPUGeneratorImpl based on input byte tensor size.
      CPUGeneratorImplStateLegacy* legacy_pod;
      auto new_state_size = new_state.numel();
      if (new_state_size == size_legacy) {
        legacy_pod = (CPUGeneratorImplStateLegacy*)new_state.data();
        // Note that in CPUGeneratorImplStateLegacy, we didn't have float version
        // of normal sample and hence we leave the c10::optional<float> as is

        // Update next_double_normal_sample.
        // Note that CPUGeneratorImplStateLegacy stores two uniform values (normal_x, normal_y)
        // and a rho value (normal_rho). These three values were redundant and in the new
        // DistributionsHelper.h, we store the actual extra normal sample, rather than three
        // intermediate values.
        if (legacy_pod->normal_is_valid) {
          auto r = legacy_pod->normal_rho;
          auto theta = 2.0 * c10::pi<double> * legacy_pod->normal_x;
          // we return the sin version of the normal sample when in caching mode
          double_normal_sample = c10::optional<double>(r * ::sin(theta));
        }
      } else if (new_state_size == size_current) {
        auto rng_state = (CPUGeneratorImplState*)new_state.data();
        legacy_pod = &rng_state->legacy_pod;
        // update next_float_normal_sample
        if (rng_state->is_next_float_normal_sample_valid) {
          float_normal_sample = c10::optional<float>(rng_state->next_float_normal_sample);
        }

        // Update next_double_normal_sample.
        // Note that in getRNGState, we now return the actual normal sample in normal_y
        // and if it's valid in normal_is_valid. The redundant normal_x and normal_rho
        // are squashed to 0.0.
        if (legacy_pod->normal_is_valid) {
          double_normal_sample = c10::optional<double>(legacy_pod->normal_y);
        }
      } else {
        AT_ERROR("Expected either a CPUGeneratorImplStateLegacy of size ", size_legacy,
                 " or a CPUGeneratorImplState of size ", size_current,
                 " but found the input RNG state size to be ", new_state_size);
      }

      // construct engine_
      // Note that CPUGeneratorImplStateLegacy stored a state array of 64 bit uints, whereas in our
      // redefined mt19937, we have changed to a state array of 32 bit uints. Hence, we are
      // doing a std::copy.
      at::mt19937_data_pod rng_data;
      std::copy(std::begin(legacy_pod->state), std::end(legacy_pod->state), rng_data.state_.begin());
      rng_data.seed_ = legacy_pod->the_initial_seed;
      rng_data.left_ = legacy_pod->left;
      rng_data.seeded_ = legacy_pod->seeded;
      rng_data.next_ = static_cast<u32>(legacy_pod->next);
      engine.set_data(rng_data);
      TORCH_CHECK(engine.is_valid(), "Invalid mt19937 state");
      this->engine_ = engine;
      this->next_float_normal_sample_ = float_normal_sample;
      this->next_double_normal_sample_ = double_normal_sample;
        */
    }

    /**
      | Gets the current internal state of CPUGeneratorImpl.
      | The internal state is returned as a CPU
      | byte tensor.
      |
      */
    pub fn get_state(&self) -> IntrusivePtr<TensorImpl> {
        
        todo!();
        /*
            using detail::CPUGeneratorImplState;

      static const usize size = sizeof(CPUGeneratorImplState);
      static_assert(std::is_pod<CPUGeneratorImplState>::value, "CPUGeneratorImplState is not a PODType");

      auto state_tensor = at::detail::empty_cpu({(i64)size}, ScalarType::Byte, c10::nullopt, c10::nullopt, c10::nullopt, c10::nullopt);
      auto rng_state = state_tensor.data_ptr();

      // accumulate generator data to be copied into byte tensor
      auto accum_state = std::make_unique<CPUGeneratorImplState>();
      auto rng_data = this->engine_.data();
      accum_state->legacy_pod.the_initial_seed = rng_data.seed_;
      accum_state->legacy_pod.left = rng_data.left_;
      accum_state->legacy_pod.seeded = rng_data.seeded_;
      accum_state->legacy_pod.next = rng_data.next_;
      std::copy(rng_data.state_.begin(), rng_data.state_.end(), std::begin(accum_state->legacy_pod.state));
      accum_state->legacy_pod.normal_x = 0.0; // we don't use it anymore and this is just a dummy
      accum_state->legacy_pod.normal_rho = 0.0; // we don't use it anymore and this is just a dummy
      accum_state->legacy_pod.normal_is_valid = false;
      accum_state->legacy_pod.normal_y = 0.0;
      accum_state->next_float_normal_sample = 0.0f;
      accum_state->is_next_float_normal_sample_valid = false;
      if (this->next_double_normal_sample_) {
        accum_state->legacy_pod.normal_is_valid = true;
        accum_state->legacy_pod.normal_y = *(this->next_double_normal_sample_);
      }
      if (this->next_float_normal_sample_) {
        accum_state->is_next_float_normal_sample_valid = true;
        accum_state->next_float_normal_sample = *(this->next_float_normal_sample_);
      }

      memcpy(rng_state, accum_state.get(), size);
      return state_tensor.getIntrusivePtr();
        */
    }

    /**
      | Gets the DeviceType of CPUGeneratorImpl.
      | 
      | Used for type checking during run time.
      |
      */
    pub fn device_type(&mut self) -> DeviceType {
        
        todo!();
        /*
            return DeviceType::CPU;
        */
    }

    /**
      | Gets a random 32 bit unsigned integer
      | from the engine
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn random(&mut self) -> u32 {
        
        todo!();
        /*
            return engine_();
        */
    }

    /**
      | Gets a random 64 bit unsigned integer
      | from the engine
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn random64(&mut self) -> u64 {
        
        todo!();
        /*
            u32 random1 = engine_();
      u32 random2 = engine_();
      return detail::make64BitsFrom32Bits(random1, random2);
        */
    }

    /**
      | Get the cached normal random in float
      |
      */
    pub fn next_float_normal_sample(&mut self) -> Option<f32> {
        
        todo!();
        /*
            return next_float_normal_sample_;
        */
    }

    /**
      | Get the cached normal random in double
      |
      */
    pub fn next_double_normal_sample(&mut self) -> Option<f64> {
        
        todo!();
        /*
            return next_double_normal_sample_;
        */
    }

    /**
      | Cache normal random in float
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn set_next_float_normal_sample(&mut self, randn: Option<f32>)  {
        
        todo!();
        /*
            next_float_normal_sample_ = randn;
        */
    }

    /**
      | Cache normal random in double
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn set_next_double_normal_sample(&mut self, randn: Option<f64>)  {
        
        todo!();
        /*
            next_double_normal_sample_ = randn;
        */
    }

    /**
      | Get the engine of the CPUGeneratorImpl
      |
      */
    pub fn engine(&mut self) -> Mt19937 {
        
        todo!();
        /*
            return engine_;
        */
    }

    /**
      | Set the engine of the CPUGeneratorImpl
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn set_engine(&mut self, engine: Mt19937)  {
        
        todo!();
        /*
            engine_ = engine;
        */
    }

    /**
      | Public clone method implementation
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn clone(&self) -> Arc<CPUGeneratorImpl> {
        
        todo!();
        /*
            return std::shared_ptr<CPUGeneratorImpl>(this->clone_impl());
        */
    }

    /**
      | Private clone method implementation
      | 
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    pub fn clone_impl(&self) -> *mut CPUGeneratorImpl {
        
        todo!();
        /*
            auto gen = new CPUGeneratorImpl();
      gen->set_engine(engine_);
      gen->set_next_float_normal_sample(next_float_normal_sample_);
      gen->set_next_double_normal_sample(next_double_normal_sample_);
      return gen;
        */
    }
}
