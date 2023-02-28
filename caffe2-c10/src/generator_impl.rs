/*!
 | Note [Generator]
 | ~~~~~~~~~~~~~~~~
 | A Pseudo Random Number Generator (PRNG) is an
 | engine that uses an algorithm to generate
 | a seemingly random sequence of numbers, that
 | may be later be used in creating a random
 | distribution. Such an engine almost always
 | maintains a state and requires a seed to start
 | off the creation of random numbers. Often
 | times, users have found it beneficial to be
 | able to explicitly create, retain, and destroy
 | PRNG states and also be able to have control
 | over the seed value.
 |
 | A Generator in ATen gives users the ability to
 | read, write and modify a PRNG engine. For
 | instance, it does so by letting users seed
 | a PRNG engine, fork the state of the engine,
 | etc.
 |
 | By default, there is one generator per device,
 | and a device's generator is lazily
 | created. A user can use the torch.Generator()
 | api to create their own generator. Currently
 | torch.Generator() can only create
 | a CPUGeneratorImpl.
 |
 | Note [Acquire lock when using random generators]
 | ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
 | Generator and its derived classes are NOT
 | thread-safe. Please note that most of the
 | places where we have inserted locking for
 | generators are historically based, and we
 | haven't actually checked that everything is
 | truly thread safe (and it probably
 | isn't). Please use the public mutex_ when using
 | any methods from these classes, except for the
 | read-only methods. You can learn about the
 | usage by looking into the unittests
 | (aten/src/ATen/cpu_generator_test.cpp) and
 | other places where we have used lock_guard.
 |
 | TODO: Look into changing the threading
 | semantics of Generators in ATen (e.g., making
 | them non-thread safe and instead making the
 | generator state splittable, to accommodate
 | forks into other threads).
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/c10/core/GeneratorImpl.h]

/**
  | Common methods for all generators
  |
  */
pub trait GeneratorImplInterface:
SetCurrentSeed
+ CurrentSeed
+ Seed
+ SetState
+ GetState 
+ CloneImpl {}

pub trait SetCurrentSeed {

    fn set_current_seed(&mut self, seed: u64);
}

pub trait CurrentSeed {

    fn current_seed(&self) -> u64;
}

pub trait Seed {

    fn seed(&mut self) -> u64;
}

pub trait SetState {

    fn set_state(&mut self, new_state: &TensorImpl);
}

pub trait GetState {

    fn get_state(&self) -> TensorImplAdapter;
}

pub trait CloneImpl {

    fn clone_impl(&self) -> *mut GeneratorImpl;
}

/**
  | The default seed is selected to be a large
  | number with good distribution of 0s and 1s in
  | bit representation
  |
  */
pub const DEFAULT_RNG_SEED_VAL: u64 = 67280421310721;

pub struct GeneratorImpl {
    link: LinkedListLink,

    /**
      | See Note [Acquire lock when using random
      | generators]
      |
      */
    mutex:   RawMutex,
    device:  Device,
    key_set: DispatchKeySet,
    pyobj:   *mut PyObject, // default = nullptr
}

intrusive_adapter!(pub GeneratorImplAdapter = Box<GeneratorImpl>: GeneratorImpl { link: LinkedListLink });

impl GeneratorImpl {
    
    pub fn key_set(&self) -> DispatchKeySet {
        
        todo!();
        /*
            return key_set_;
        */
    }
    
    #[inline] pub fn set_pyobj(&mut self, pyobj: *mut PyObject)  {
        
        todo!();
        /*
            pyobj_ = pyobj;
        */
    }
    
    #[inline] pub fn pyobj(&self) -> *mut PyObject {
        
        todo!();
        /*
            return pyobj_;
        */
    }
}

//-------------------------------------------[.cpp/pytorch/c10/core/GeneratorImpl.cpp]

impl GeneratorImpl {
    
    /**
      | GeneratorImpl class implementation
      |
      */
    pub fn new(
        device_in: Device,
        key_set:   DispatchKeySet) -> Self {
    
        todo!();
        /*


            : device_{device_in}, key_set_(key_set)
        */
    }

    /**
      | Clone this generator. Note that clone()
      | is the only method for copying for Generators
      | in ATen.
      |
      */
    pub fn clone(&self) -> GeneratorImplAdapter {
        
        todo!();
        /*
            auto res = this->clone_impl();
      raw::intrusive_ptr::incref(res);
      return intrusive_ptr<GeneratorImpl>::reclaim(res);
        */
    }

    /**
      | Gets the device of a generator.
      |
      */
    pub fn device(&self) -> Device {
        
        todo!();
        /*
            return device_;
        */
    }
}

/**
  | Gets a random number for /dev/urandom
  | 
  | Note this is a legacy method (from THRandom.cpp)
  | 
  | FIXME: use random_device with entropy
  | information
  |
  */
#[cfg(not(_WIN32))]
pub fn read_urandom_long() -> u64 {
    
    todo!();
        /*
            int randDev = open("/dev/urandom", O_RDONLY);
      // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      uint64_t randValue;
      TORCH_CHECK(randDev >= 0, "Unable to open /dev/urandom");
      ssize_t readBytes = read(randDev, &randValue, sizeof(randValue));
      TORCH_CHECK(
          readBytes >= (ssize_t)sizeof(randValue),
          "Unable to read from /dev/urandom");
      close(randDev);
      return randValue;
        */
}

/**
 | Gets a non deterministic random number number
 | from either the /dev/urandom or the current
 | time. 
 |
 | For Cuda, gets random from random_device and
 | adds a transformation on it.
 |
 | FIXME: The behavior in this function is from
 | legacy code
 |
 | (THRandom_seed/THCRandom_seed) and is probably
 | not the right thing to do, even though our
 | tests pass. Figure out if tests get perturbed
 |
 | - when the same algorithm is used for all
 | backends. Note that the current behavior is
 | different for CPU, Cuda and Windows CPU.
 |
 | - when using C++11 std objects, such as
 | random_device
 |
 | - when constructing a 64 bit seed properly,
 |   rather than static casting a 32 bit number to
 |   64 bit.
 */
pub fn get_non_deterministic_random(is_cuda: Option<bool>) -> u64 {

    let is_cuda: bool = is_cuda.unwrap_or(false);
    
    todo!();
        /*
            // NOLINTNEXTLINE(cppcoreguidelines-init-variables)
      uint64_t s;
      if (!is_cuda) {
    #ifdef _WIN32
        s = (uint64_t)chrono::high_resolution_clock::now()
                .time_since_epoch()
                .count();
    #else
        s = readURandomLong();
    #endif
      } else {
        random_device rd;
        // limit to 53 bits to ensure unique representation in double
        s = ((((uint64_t)rd()) << 32) + rd()) & 0x1FFFFFFFFFFFFF;
      }
      return s;
        */
}
