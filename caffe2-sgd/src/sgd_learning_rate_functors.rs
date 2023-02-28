crate::ix!();

/**
  | LearningRateFunctor is a functor that
  | when fed with an iter number, produces
  | the learning rate for the corresponding
  | iteration.
  |
  */
pub trait LearningRateFunctor<T> {
    fn invoke(&self, iter: i64) -> T;
}

/**
  | Fixed: not changing the learning rate
  | at all.
  |
  */
pub struct FixedLearningRate<T> { 
    phantom: PhantomData<T>,
}

impl<T> LearningRateFunctor<T> for FixedLearningRate<T> {
    
    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            return 1.;
        */
    }
}

/**
  | Alter: alternatate learning rate with
  | active_period and inactive_period.
  | update for for a duration of active_period
  | and then stop for a duration of inactive_period
  | if active_first, and vice versa
  |
  */
pub struct AlternateLearningRate<T> {
    active_period:    i64,
    inactive_period:  i64,
    active_first:     bool,
    phantom: PhantomData<T>,
}

impl<T> LearningRateFunctor<T> for AlternateLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter % (active_period_ + inactive_period_) <
            (active_first_ ? active_period_ : inactive_period_)) {
          return active_first_ ? 1. : 0.;
        } else {
          return active_first_ ? 0. : 1.;
        };
        */
    }
}

impl<T> AlternateLearningRate<T> {

    pub fn new(
        active_period:   i64,
        inactive_period: i64,
        active_first:    bool) -> Self {
    
        todo!();
        /*
            : active_period_(active_period),
            inactive_period_(inactive_period),
            active_first_(active_first)
        */
    }
}

/**
  | Step: return gamma ^ (floor(iter / step))
  |
  */
pub struct StepLearningRate<T> {
    stepsize:  i32,
    gamma:     T,
}

impl<T> LearningRateFunctor<T> for StepLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            return std::pow(gamma_, static_cast<T>(iter / stepsize_));
        */
    }
}

impl<T> StepLearningRate<T> {

    pub fn new(stepsize: i32, gamma: T) -> Self {
    
        todo!();
        /*
            : stepsize_(stepsize), gamma_(gamma)
        */
    }
}

/**
  | Exp: return gamma ^ iter
  |
  */
pub struct ExpLearningRate<T> {
    gamma:  T,
}

impl<T> LearningRateFunctor<T> for ExpLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            return std::pow(gamma_, static_cast<T>(iter));
        */
    }
}

impl<T> ExpLearningRate<T> {

    pub fn new(gamma: T) -> Self {
    
        todo!();
        /*
            : gamma_(gamma)
        */
    }
}

/**
  | Gate: return multiplier_1 if before
  | num_iter, else multiplier_2
  |
  */
pub struct GateLearningRate<T> {
    multiplier_1:  T,
    multiplier_2:  T,
    num_iter:      u64,
}

impl<T> LearningRateFunctor<T> for GateLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {

        todo!();
        /*
           if (iter >= int64_t(num_iter_)) {
           return T(multiplier_2_);
           }
           return T(multiplier_1_);
           */
    }
}

impl<T> GateLearningRate<T> {

    pub fn new(
        multiplier_1: T,
        multiplier_2: T,
        num_iter:     i64) -> Self {

        todo!();
        /*
           : multiplier_1_(multiplier_1),
           multiplier_2_(multiplier_2),
           num_iter_(num_iter)
           */
    }
}

/**
  | Inv: return (1 + gamma * iter) ^ (-power)
  |
  */
pub struct InvLearningRate<T> {
    gamma:  T,
    power:  T,
}

impl<T> LearningRateFunctor<T> for InvLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            return std::pow(T(1) + gamma_ * iter, -power_);
        */
    }
}

impl<T> InvLearningRate<T> {

    pub fn new(gamma: T, power: T) -> Self {
    
        todo!();
        /*
            : gamma_(gamma), power_(power)
        */
    }
}

/**
  | Poly: return (1 - iter/max_iter) ^ (power)
  |
  */
pub struct PolyLearningRate<T> {
    power:     T,
    max_iter:  u64,
}

impl<T> LearningRateFunctor<T> for PolyLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            return std::pow(1 - T(iter) / T(max_iter_), power_);
        */
    }
}

impl<T> PolyLearningRate<T> {

    pub fn new(power: T, max_iter: i64) -> Self {
    
        todo!();
        /*
            : power_(power), max_iter_(max_iter)
        */
    }
}

/**
  | LinearWarmup: return max(iter/num_iter,
  | 1)
  |
  */
pub struct LinearWarmupLearningRate<T> {
    start_multiplier:  T,
    num_iter:          u64,
}

impl<T> LearningRateFunctor<T> for LinearWarmupLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter >= int64_t(num_iter_)) {
          return 1.;
        }
        return start_multiplier_ +
            (1. - start_multiplier_) * T(iter) / T(num_iter_);
        */
    }
}

impl<T> LinearWarmupLearningRate<T> {

    pub fn new(start_multiplier: T, num_iter: i64) -> Self {
    
        todo!();
        /*
            : start_multiplier_(start_multiplier), num_iter_(num_iter)
        */
    }
}

/**
  | ConstantWarmup: return scale when
  | iter < num_iter, and 1 otherwise
  |
  */
pub struct ConstantWarmupLearningRate<T> {
    multiplier:  T,
    num_iter:    u64,
}

impl<T> LearningRateFunctor<T> for ConstantWarmupLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter >= int64_t(num_iter_)) {
          return 1.;
        }
        return T(multiplier_);
        */
    }
}

impl<T> ConstantWarmupLearningRate<T> {

    pub fn new(multiplier: T, num_iter: i64) -> Self {
    
        todo!();
        /*
            : multiplier_(multiplier), num_iter_(num_iter)
        */
    }
}

/**
  | ConstantWarmup: return scale when
  | iter < num_iter, and 1 otherwise
  |
  */
pub struct PieceWarmupLearningRate<T> {
    m1:  T,
    m2:  T,
    m3:  T,
    n1:  u64,
    n2:  u64,
}

impl<T> LearningRateFunctor<T> for PieceWarmupLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter < int64_t(n1_)) {
          return m1_;
        } else if (iter < int64_t(n2_)) {
          return m2_;
        }
        return m3_;
        */
    }
}

impl<T> PieceWarmupLearningRate<T> {

    pub fn new(
        m1: T,
        n1: i64,
        m2: T,
        n2: i64,
        m3: T) -> Self {
    
        todo!();
        /*
            : m1_(m1), m2_(m2), m3_(m3), n1_(n1), n2_(n2)
        */
    }
}

/**
  | hill: the learning rate changes according
  | to following 3 stages
  | 
  | 1) linear warmup (increasing) at first
  | num_iter steps from start_multiplier
  | 
  | 2) inverse shrink (decreasing) afterwards
  | (gamma, power)
  | 
  | 3) lower bounded by end_multiplier
  |
  */
pub struct HillLearningRate<T> {
    linear_warmup_lr:  LinearWarmupLearningRate<T>,
    inv_lr:            InvLearningRate<T>,
    num_iter:          i64,
    end_multiplier:    T,
}

impl<T> LearningRateFunctor<T> for HillLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter < num_iter_) {
          return linear_warmup_lr_(iter);
        } else {
          return std::max(end_multiplier_, inv_lr_(iter - num_iter_));
        }
        */
    }
}

impl<T> HillLearningRate<T> {

    pub fn new(
        num_iter:         i64,
        start_multiplier: T,
        gamma:            T,
        power:            T,
        end_multiplier:   T) -> Self {
    
        todo!();
        /*
            : linear_warmup_lr_(start_multiplier, num_iter),
            inv_lr_(gamma, power),
            num_iter_(num_iter),
            end_multiplier_(end_multiplier)
        */
    }
}

/**
 |
 | slope: the learning rate changes according to
 | 2 stages
 |
 | -1) constantWarmup with multiplier_1
 | -2) linearly shink to multiplier_2:
 |
 |  max{
 |
 |     multiplier_1 + (iter - num_iter_1)
 |     * (multiplier_2 - multiplier_1) / (num_iter_2
 |     - num_iter_1),
 |
 |     multiplier_2
 |
 |  }
 */
pub struct SlopeLearningRate<T> {
    num_iter_1:    i64,
    multiplier_1:  T,
    num_iter_2:    i64,
    multiplier_2:  T,
}

impl<T> LearningRateFunctor<T> for SlopeLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter < num_iter_1_) {
          return multiplier_1_;
        } else {
          return std::max(
            multiplier_2_,
            multiplier_1_ + (iter - num_iter_1_) * (multiplier_2_ - multiplier_1_) / (num_iter_2_ - num_iter_1_)
          );
        }
        */
    }
}

impl<T> SlopeLearningRate<T> {

    pub fn new(
        num_iter_1:   i64,
        multiplier_1: T,
        num_iter_2:   T,
        multiplier_2: T) -> Self {

        todo!();
        /*
            : num_iter_1_(num_iter_1),
            multiplier_1_(multiplier_1),
            num_iter_2_(num_iter_2),
            multiplier_2_(multiplier_2)
        */
    }
}

pub struct CompositeLearningRateItem<T> {
    num_iter:  i64,
    lr_scale:  f32,
    policy:    *mut dyn LearningRateFunctor<T>,
}

impl<T> CompositeLearningRateItem<T> {

    pub fn new(
        num_iter: i64,
        lr_scale: f32,
        policy:   *mut dyn LearningRateFunctor<T>) -> Self {

        todo!();
        /*
            : num_iter_(num_iter), lr_scale_(lr_scale), policy_(policy)
        */
    }
}

/**
  | composite: the learning policy changes
  | according to current iteration #
  |
  */
pub struct CompositeLearningRate<T> {
    sub_policies:          HashMap<i64,Box<dyn LearningRateFunctor<T>>>,
    sub_policy_lr_scales:  HashMap<i64,f32>,
}

impl<T> LearningRateFunctor<T> for CompositeLearningRate<T> {
    
    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            auto sub_policy = sub_policies_.upper_bound(iter);
        DCHECK(sub_policy != sub_policies_.begin());
        --sub_policy;
        auto sub_policy_lr_scale = sub_policy_lr_scales_.upper_bound(iter);
        DCHECK(sub_policy_lr_scale != sub_policy_lr_scales_.begin());
        --sub_policy_lr_scale;
        return ((*sub_policy->second)(iter)) * (sub_policy_lr_scale->second);
        */
    }
}

impl<T> CompositeLearningRate<T> {

    pub fn new(sub_policies: &LinkedList<CompositeLearningRateItem<T>>) -> Self {
    
        todo!();
        /*
            DCHECK_GT(sub_policies.size(), 0);
        int64_t num_iter_start = 1;
        for (auto it = sub_policies.begin(); it != sub_policies.end(); ++it) {
          DCHECK_GT(it->num_iter_, 0);
          sub_policies_[num_iter_start].reset(it->policy_);
          sub_policy_lr_scales_[num_iter_start] = it->lr_scale_;
          num_iter_start += it->num_iter_;
        }
        */
    }
}

/**
  | Cyclical: return a learning rate with
  | period 2 * stepsize and lower bound base_lr,
  | upper bound max_lr.
  | 
  | See https://arxiv.org/pdf/1506.01186.pdf
  |
  */
pub struct CyclicalLearningRate<T> {
    base_lr:   T,
    max_lr:    T,
    stepsize:  i32,
    decay:     T,
}

impl<T> LearningRateFunctor<T> for CyclicalLearningRate<T> {
    
    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            int64_t cycle = static_cast<int>((iter / (2 * stepsize_)) + 1);
        T x = abs(static_cast<T>(iter) / stepsize_ - 2 * cycle + 1);
        return 1 +
            (T(abs(max_lr_)) / T(abs(base_lr_)) - 1) * std::max(T(0.0), (1 - x)) *
            std::pow(decay_, static_cast<int>(iter / (2 * stepsize_)));
        */
    }
}

impl<T> CyclicalLearningRate<T> {

    pub fn new(
        base_lr:  T,
        max_lr:   T,
        stepsize: i32,
        decay:    T) -> Self {
    
        todo!();
        /*
            : base_lr_(base_lr),
            max_lr_(max_lr),
            stepsize_(stepsize),
            decay_(decay)
        */
    }
}

/**
  | Cosine: return a learning rate with
  | a cosine schedule lower bound min_lr,
  | upper bound max_lr.
  | 
  | See https://arxiv.org/pdf/1608.03983.pdf
  |
  */
pub struct CosineLearningRate<T> {
    min_lr:     T,
    max_lr:     T,
    period:     i64,
    t_mult:     T,
    lr_shrink:  T,
}

impl<T> LearningRateFunctor<T> for CosineLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            T i, t_i, t_curr;
        if (t_mult_ != 1.0) {
          // the period is changed every time
          i = floor(
              log(1 - double(iter) / double(period_) * (1.0 - t_mult_)) /
              log(t_mult_));
          t_i = pow(t_mult_, i) * period_;
          t_curr = iter - (1.0 - pow(t_mult_, i)) / (1.0 - t_mult_) * period_;
        } else {
          // fixed period
          i = floor(double(iter) / double(period_));
          t_i = period_;
          t_curr = iter - t_i * i;
        }
        T lr_shrink = pow(lr_shrink_, i);
        T min_lr = min_lr_ * lr_shrink;
        T max_lr = max_lr_ * lr_shrink;
        T final_lr =
            min_lr + 0.5 * (max_lr - min_lr) * (1 + cos(M_PI * t_curr / t_i));
        return final_lr;
        */
    }
}

impl<T> CosineLearningRate<T> {

    pub fn new(
        min_lr:    T,
        max_lr:    T,
        period:    i64,
        t_mult:    T,
        lr_shrink: T) -> Self {

        todo!();
        /*
            : min_lr_(min_lr),
            max_lr_(max_lr),
            period_(period),
            t_mult_(t_mult),
            lr_shrink_(lr_shrink)
        */
    }
}

/**
  | constantThenLinearWarmup: first
  | use a constant multiplier and then ramp
  | up to the global lr
  |
  */
pub struct ConstantThenLinearWarmupLearningRate<T> {
    constant_warmup_num_iter:  i64,
    linear_warmup_num_iter:    i64,
    constant_warmup_lr:        ConstantWarmupLearningRate<T>,
    linear_warmup_lr:          LinearWarmupLearningRate<T>,
}

impl<T> LearningRateFunctor<T> for ConstantThenLinearWarmupLearningRate<T> {
    
    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter < constant_warmup_num_iter_) {
          return constant_warmup_lr_(iter);
        } else if (iter < constant_warmup_num_iter_ + linear_warmup_num_iter_) {
          return linear_warmup_lr_(iter - constant_warmup_num_iter_);
        } else {
          return 1.0;
        }
        */
    }
}

impl<T> ConstantThenLinearWarmupLearningRate<T> {

    pub fn new(
        start_warmup_multiplier:  T,
        constant_warmup_num_iter: i64,
        linear_warmup_num_iter:   i64) -> Self {

        todo!();
        /*
            : constant_warmup_num_iter_(constant_warmup_num_iter),
            linear_warmup_num_iter_(linear_warmup_num_iter),
            constant_warmup_lr_(start_warmup_multiplier, constant_warmup_num_iter),
            linear_warmup_lr_(start_warmup_multiplier, linear_warmup_num_iter)
        */
    }
}

///------------------------
/**
  CompositeCosineLearningRate: first use a constant multiplier
  and then ramp up to the global lr, and then use a cosine learning rate
  */
pub struct CompositeCosineLearningRate<T> {
    constant_warmup_num_iter:        i64,
    linear_warmup_num_iter:          i64,
    constant_then_linear_warmup_lr:  ConstantThenLinearWarmupLearningRate<T>,
    cosine_lr:                       CosineLearningRate<T>,
}

impl<T> LearningRateFunctor<T> for CompositeCosineLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter < constant_warmup_num_iter_ + linear_warmup_num_iter_) {
          return constant_then_linear_warmup_lr_(iter);
        }
        return cosine_lr_(
            iter - constant_warmup_num_iter_ - linear_warmup_num_iter_);
        */
    }
}

impl<T> CompositeCosineLearningRate<T> {

    pub fn new(
        start_warmup_multiplier:  T,
        constant_warmup_num_iter: i64,
        linear_warmup_num_iter:   i64,
        cosine_min_lr:            T,
        cosine_max_lr:            T,
        cosine_period:            i64,
        consine_t_mult:           T,
        cosine_lr_shrink:         T) -> Self {
    
        todo!();
        /*
            : constant_warmup_num_iter_(constant_warmup_num_iter),
            linear_warmup_num_iter_(linear_warmup_num_iter),
            constant_then_linear_warmup_lr_(
                start_warmup_multiplier,
                constant_warmup_num_iter,
                linear_warmup_num_iter),
            cosine_lr_(
                cosine_min_lr,
                cosine_max_lr,
                cosine_period,
                consine_t_mult,
                cosine_lr_shrink)
        */
    }
}

///----------------------------------------
/**
  CompositeCyclicalLearningRate: first use a constant multiplier
  and then ramp up to the global lr, and then use a cyclical learning rate
  */
pub struct CompositeCyclicalLearningRate<T> {
    constant_warmup_num_iter:        i64,
    linear_warmup_num_iter:          i64,
    constant_then_linear_warmup_lr:  ConstantThenLinearWarmupLearningRate<T>,
    cyclical_lr:                     CyclicalLearningRate<T>,
}

impl<T> LearningRateFunctor<T> for CompositeCyclicalLearningRate<T> {

    #[inline] fn invoke(&self, iter: i64) -> T {
        
        todo!();
        /*
            if (iter < constant_warmup_num_iter_ + linear_warmup_num_iter_) {
          return constant_then_linear_warmup_lr_(iter);
        }
        return cyclical_lr_(
            iter - constant_warmup_num_iter_ - linear_warmup_num_iter_);
        */
    }
}

impl<T> CompositeCyclicalLearningRate<T> {

    pub fn new(
        base_lr:                  T,
        start_warmup_multiplier:  T,
        constant_warmup_num_iter: i64,
        linear_warmup_num_iter:   i64,
        cyclical_max_lr:          T,
        cyclical_step_size:       i32,
        cyclical_decay:           T) -> Self {

        todo!();
        /*
            : constant_warmup_num_iter_(constant_warmup_num_iter),
            linear_warmup_num_iter_(linear_warmup_num_iter),
            constant_then_linear_warmup_lr_(
                start_warmup_multiplier,
                constant_warmup_num_iter,
                linear_warmup_num_iter),
            cyclical_lr_(
                base_lr,
                cyclical_max_lr,
                cyclical_step_size,
                cyclical_decay)
        */
    }
}
