/*!
 | Distributions kernel adapted from THRandom.cpp
 | The kernels try to follow random distributions signature
 | For instance: in ATen
 |      auto gen = createCPUGenerator();
 |      uniform_real_distribution<double> uniform(0, 1);
 |      auto sample = uniform(gen.get());
 |
 |      vs random
 |
 |      mt19937 gen;
 |      uniform_real_distribution uniform(0, 1);
 |      auto sample = uniform(gen);
 */

crate::ix!();

//-------------------------------------------[.cpp/pytorch/aten/src/ATen/core/DistributionsHelper.h]

/**
  | Samples a discrete uniform distribution
  | in the range [base, base+range) of type
  | T
  |
  */
pub struct UniformIntFromToDistribution<T> {
    range: u64,
    base:  i64,
}

impl<T> UniformIntFromToDistribution<T> {

     
    pub fn new(
        range: u64,
        base:  i64) -> Self {
    
        todo!();
        /*


            range_ = range;
        base_ = base;
        */
    }

     
    #[inline] pub fn invoke(&mut self, generator: RNG) -> T {
        
        todo!();
        /*
            if ((
          is_same<T, i64>::value ||
          is_same<T, double>::value ||
          is_same<T, float>::value ||
          is_same<T, BFloat16>::value) && range_ >= 1ULL << 32)
        {
          return transformation::uniform_int_from_to<T>(generator->random64(), range_, base_);
        } else {
          return transformation::uniform_int_from_to<T>(generator->random(), range_, base_);
        }
        */
    }
}

/**
  | Samples a discrete uniform distribution
  | in the range [min_value(i64),
  | max_value(i64)]
  |
  */
pub struct UniformIntFullRangeDistribution<T> {

}

impl<T> UniformIntFullRangeDistribution<T> {

     
    #[inline] pub fn invoke(&mut self, generator: RNG) -> T {
        
        todo!();
        /*
            return transformation::uniform_int_full_range<T>(generator->random64());
        */
    }
}

/**
  | Samples a discrete uniform distribution
  | in the range [0, max_value(T)] for integral
  | types and [0, 2^mantissa] for floating-point
  | types.
  |
  */
pub struct UniformIntDistribution<T> {

}

impl<T> UniformIntDistribution<T> {

    #[inline] pub fn invoke(&mut self, generator: RNG) -> T {
        
        todo!();
        /*
            if (is_same<T, double>::value || is_same<T, i64>::value) {
          return transformation::uniform_int<T>(generator->random64());
        } else {
          return transformation::uniform_int<T>(generator->random());
        }
        */
    }
}

/**
  | Samples a uniform distribution in the
  | range [from, to) of type T
  |
  */
pub struct UniformRealDistribution<T> {
    from: T,
    to:   T,
}

impl<T> UniformRealDistribution<T> {
     
    pub fn new(from: T, to: T) -> Self {
    
        todo!();
        /*


            TORCH_CHECK_IF_NOT_ON_CUDA(from <= to);
        TORCH_CHECK_IF_NOT_ON_CUDA(to - from <= T::max);
        from_ = from;
        to_ = to;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> DistAccType<T> {
    
        todo!();
        /*
            if(is_same<T, double>::value) {
          return transformation::uniform_real<T>(generator->random64(), from_, to_);
        } else {
          return transformation::uniform_real<T>(generator->random(), from_, to_);
        }
        */
    }
}

/**
  | The SFINAE checks introduced in #39816
  | looks overcomplicated and must revisited
  | https://github.com/pytorch/pytorch/issues/40052
  |
  */
#[macro_export] macro_rules! distribution_helper_generate_has_member {
    ($member:ident) => {
        /*
        
        template <typename T>                                                
        struct has_member_##member                                           
        {                                                                    
            typedef char yes;                                                
            typedef long no;                                                 
            template <typename U> static yes test(decltype(&U::member));     
            template <typename U> static no test(...);                       
            static constexpr bool value = sizeof(test<T>(0)) == sizeof(yes); 
        }
        */
    }
}

distribution_helper_generate_has_member!(next_double_normal_sample);
distribution_helper_generate_has_member!(set_next_double_normal_sample);
distribution_helper_generate_has_member!(next_float_normal_sample);
distribution_helper_generate_has_member!(set_next_float_normal_sample);

#[macro_export] macro_rules! distribution_helper_generate_next_normal_methods {
    ($TYPE:ident) => {
        /*
        
                                                                                                            
        template <typename RNG, typename ret_type,                                                          
                  typename enable_if_t<(                                                               
                    has_member_next_##TYPE##_normal_sample<RNG>::value &&                                   
                    has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  
                  ), int> = 0>                                                                              
        inline bool maybe_get_next_##TYPE##_normal_sample(RNG* generator, ret_type* ret) {  
          if (generator->next_##TYPE##_normal_sample()) {                                                   
            *ret = *(generator->next_##TYPE##_normal_sample());                                             
            generator->set_next_##TYPE##_normal_sample(optional<TYPE>());                              
            return true;                                                                                    
          }                                                                                                 
          return false;                                                                                     
        }                                                                                                   
                                                                                                            
        template <typename RNG, typename ret_type,                                                          
                  typename enable_if_t<(                                                               
                    !has_member_next_##TYPE##_normal_sample<RNG>::value ||                                  
                    !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 
                  ), int> = 0>                                                                              
        C10_HOST_DEVICE inline bool maybe_get_next_##TYPE##_normal_sample(RNG* generator, ret_type* ret) {  
          return false;                                                                                     
        }                                                                                                   
                                                                                                            
        template <typename RNG, typename ret_type,                                                          
                  typename enable_if_t<(                                                               
                    has_member_set_next_##TYPE##_normal_sample<RNG>::value                                  
                  ), int> = 0>                                                                              
        inline void maybe_set_next_##TYPE##_normal_sample(RNG* generator, ret_type cache) { 
          generator->set_next_##TYPE##_normal_sample(cache);                                                
        }                                                                                                   
                                                                                                            
        template <typename RNG, typename ret_type,                                                          
                  typename enable_if_t<(                                                               
                    !has_member_set_next_##TYPE##_normal_sample<RNG>::value                                 
                  ), int> = 0>                                                                              
        inline void maybe_set_next_##TYPE##_normal_sample(RNG* generator, ret_type cache) { 
        }
        */
    }
}

distribution_helper_generate_next_normal_methods!(double);
distribution_helper_generate_next_normal_methods!(float);

/**
  | Samples a normal distribution using
  | the Box-Muller method
  | 
  | Takes mean and standard deviation as
  | inputs
  | 
  | -----------
  | @note
  | 
  | Box-muller method returns two samples
  | at a time.
  | 
  | Hence, we cache the "next" sample in
  | the CPUGeneratorImpl class.
  |
  */
pub struct NormalDistribution<T> {
    mean: T,
    stdv: T,
}

impl<T> NormalDistribution<T> {

    pub fn new(
        mean_in: T,
        stdv_in: T) -> Self {
    
        todo!();
        /*


            TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in >= 0, "stdv_in must be positive: ", stdv_in);
        mean = mean_in;
        stdv = stdv_in;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> DistAccType<T> {
    
        todo!();
        /*
            dist_acctype<T> ret;
        // return cached values if available
        if (is_same<T, double>::value) {
          if (maybe_get_next_double_normal_sample(generator, &ret)) {
            return transformation::normal(ret, mean, stdv);
          }
        } else {
          if (maybe_get_next_float_normal_sample(generator, &ret)) {
            return transformation::normal(ret, mean, stdv);
          }
        }
        // otherwise generate new normal values
        uniform_real_distribution<T> uniform(0.0, 1.0);
        const dist_acctype<T> u1 = uniform(generator);
        const dist_acctype<T> u2 = uniform(generator);
        const dist_acctype<T> r = ::sqrt(static_cast<T>(-2.0) * ::log(static_cast<T>(1.0)-u2));
        const dist_acctype<T> theta = static_cast<T>(2.0) * pi<T> * u1;
        if (is_same<T, double>::value) {
          maybe_set_next_double_normal_sample(generator, r * ::sin(theta));
        } else {
          maybe_set_next_float_normal_sample(generator, r * ::sin(theta));
        }
        ret = r * ::cos(theta);
        return transformation::normal(ret, mean, stdv);
        */
    }
}

lazy_static!{
    /*
    template <typename T>
    struct DiscreteDistributionType { 

        using type = float; 
    };

    template <> struct DiscreteDistributionType<double> { using type = double; };
    */
}

/**
  | Samples a bernoulli distribution given
  | a probability input
  |
  */
pub struct BernoulliDistribution<T> {
    p: T,
}

impl<T> BernoulliDistribution<T> {

    pub fn new(p_in: T) -> Self {
    
        todo!();
        /*


            TORCH_CHECK_IF_NOT_ON_CUDA(p_in >= 0 && p_in <= 1);
        p = p_in;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> T {
    
        todo!();
        /*
            uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::bernoulli<T>(uniform(generator), p);
        */
    }
}


/**
  | Samples a geometric distribution given
  | a probability input
  |
  */
pub struct GeometricDistribution<T> {
    p: T,
}

impl<T> GeometricDistribution<T> {

    pub fn new(p_in: T) -> Self {
    
        todo!();
        /*


            TORCH_CHECK_IF_NOT_ON_CUDA(p_in > 0 && p_in < 1);
        p = p_in;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> T {
    
        todo!();
        /*
            uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::geometric<T>(uniform(generator), p);
        */
    }
}

/**
  | Samples an exponential distribution
  | given a lambda input
  |
  */
pub struct ExponentialDistribution<T> {
    lambda: T,
}

impl<T> ExponentialDistribution<T> {

    pub fn new(lambda_in: T) -> Self {
    
        todo!();
        /*


            lambda = lambda_in;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> T {
    
        todo!();
        /*
            uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::exponential<T>(uniform(generator), lambda);
        */
    }
}

/**
  | Samples a cauchy distribution given
  | median and sigma as inputs
  |
  */
pub struct CauchyDistribution<T> {
    median: T,
    sigma:  T,
}

impl<T> CauchyDistribution<T> {

    pub fn new(
        median_in: T,
        sigma_in:  T) -> Self {
    
        todo!();
        /*


            median = median_in;
        sigma = sigma_in;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> T {
    
        todo!();
        /*
            uniform_real_distribution<T> uniform(0.0, 1.0);
        return transformation::cauchy<T>(uniform(generator), median, sigma);
        */
    }
}

/**
  | Samples a lognormal distribution
  | 
  | Takes mean and standard deviation as
  | inputs
  | 
  | Outputs two samples at a time
  |
  */
pub struct LognormalDistribution<T> {
    mean: T,
    stdv: T,
}

impl<T> LognormalDistribution<T> {

    pub fn new(
        mean_in: T,
        stdv_in: T) -> Self {
    
        todo!();
        /*


            TORCH_CHECK_IF_NOT_ON_CUDA(stdv_in > 0);
        mean = mean_in;
        stdv = stdv_in;
        */
    }

    #[inline] pub fn invoke<RNG>(&mut self, generator: RNG) -> T {
    
        todo!();
        /*
            normal_distribution<T> normal(mean, stdv);
        return transformation::log_normal<T>(normal(generator));
        */
    }
}
