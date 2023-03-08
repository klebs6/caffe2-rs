crate::ix!();

use crate::{
    Operator,
    LearningRateFunctor,
    CPUContext
};

register_cuda_operator!{LearningRate, LearningRateOp<f32, CUDAContext>}

register_cpu_operator!{LearningRate, LearningRateOp<f32, CPUContext>}

no_gradient!{LearningRate}

pub type LearningRateOpFloatCPU = LearningRateOp<f32, CPUContext>;

/**
 | Learning rate is a decreasing function of
 | time. With low learning rates the improvements
 | will be linear. With high learning rates they will
 | start to look more exponential. Learning rate is
 | controlled by the following arguments:
 |
 |
 | Required:
 |  `iterations`
 |  `base_lr`: base learning rate
 |  `policy`: this controls how the learning rate is applied, options are:
 |    `fixed`
 |    `step`: uses `stepsize`, `gamma`
 |    `exp`: uses `gamma`
 |    `gate`: uses 'multiplier_1', 'multiplier_2', `num_iter``
 |    `inv`: uses `gamma`, `power`
 |    `linearWarmup`: uses `start_multiplier`, `num_iter`
 |    `constantWarmup`: uses `multiplier`, `num_iter`
 |    `alter`: uses  `active_first`, `active_period`, `inactive_period`
 |    `hill`: uses those in both `linearWarmup` and `inv`, plus `end_multiplier`
 |    `composite`: uses `sub_policy_num_iters` and additional args with format
 |    `cyclic`: uses `max_lr`, `stepsize`
 |    `cosine`: uses `min_lr`, `max_lr`, `period`, `t_mult`, `lr_shrink`
 |    `constantThenLinearWarmup`: uses `start_warmup_multiplier`, `constant_warmup_num_iter`, `linear_warmup_num_iter`
 |    `compositeCyclical`: uses `start_warmup_multiplier`, `constant_warmup_num_iter`, `linear_warmup_num_iter`, `cyclical_max_lr`, `cyclical_step_size`, `cyclical_decay`
 |    `compositeCosine`: uses `start_warmup_multiplier`, `constant_warmup_num_iter`, `linear_warmup_num_iter`, `cosine_max_lr`, `cosine_period`, `cosine_t_mult`, `cosine_lr_shrink`
 |    sub_policy_{sub_policy_index}_{sub_policy_arg}, for example:
 |    sub_policy_0_policy: "exp", sub_policy_0_gamma: 0.99,
 |    sub_policy_0_lr_scale: 1.2
 |    sub_policy_0_policy: "fixed", sub_policy_0_lr_scale: 1.0
 |    sub_policy_num_iters: [1000, 1000]
 |
 | Optional:
 |   `stepsize`: defaults to 0
 |   `max_lr`: defaults to 0.005
 |   `gamma`: defaults to 0
 |   `power`: defaults to 0
 |   `num_iter`: defaults to 0
 |   `start_multiplier`: defaults to 0
 |   `multiplier`: defaults to 0.5
 |   `multiplier_1`: defaults to 1
 |   `multiplier_2`: defaults to 1
 |   `m1`: defaults to 0.5, the first piece lr of piece warmup
 |   `n1`: defaults to 0, iter threshold of the first piece lr
 |   `m2`: defaults to 0.5, the second piece lr of piece warmup
 |   `n2`: defaults to 0, iter threshold of the second piece lr
 |   `m3`: defaults to 0.5, the third piece lr of piece warmup
 |   `start_warmup_multiplier`: defaults to 0.1, part of constantThenLinearWarmup
 |   `constant_warmup_num_iter`: defaults to 10000000, part of constantThenLinearWarmup and constantThenLinearWarmup
 |   `linear_warmup_num_iter`: defaults to 10000000, part of constantThenLinearWarmup, CompositeCyclicalLRPolicy, CompositeCosineLRPolicy
 |   `cyclical_max_lr`: defaults to 0.05, part of CompositeCyclicalLRPolicy
 |   `cyclical_step_size`: defaults to 1000000, part of CompositeCyclicalLRPolicy
 |   `cyclical_decay`: defaults to 1.0, part of CompositeCyclicalLRPolicy
 |   `cosine_min_lr`:defaults to 0.01, part of CompositeCosineLRPolicy
 |   `cosine_max_lr`:defaults to 0.05, part of CompositeCosineLRPolicy
 |   `cosine_period`:defaults to 50, part of CompositeCosineLRPolicy
 |   `cosine_t_mult`:defaults to 1.0, part of CompositeCosineLRPolicy
 |   `cosine_lr_shrink`:defaults to 0.99, part of CompositeCosineLRPolicy
 |
 | Usage:
 |   train_net.LearningRate(*iterations*, "*label*", base_lr=*float*,
 |                          policy="policy_name", stepsize=*int*, gamma=*float*)
 |
 |
 | Example usage:
 |   train_net.LearningRate(200, "LR", base_lr=-0.1,
 |                          policy="step", stepsize=20, gamma=0.9)
 |
 */
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct LearningRateOp<T,Context> {
    context: Context,
    functor: Box<dyn LearningRateFunctor<T>>,
    base_lr: T,
}

impl<T,Context> Operator for LearningRateOp<T,Context> {

}

num_inputs!{LearningRate, 1}

num_outputs!{LearningRate, 1}

inputs!{LearningRate, 
    0 => ("input", "description needed")
}

outputs!{LearningRate, 
    0 => ("output", "description needed")
}

args!{LearningRate, 
    0  => ("base_lr",                      "(float, required) base learning rate"),
    1  => ("policy",                       "(float, default 1.0) strategy for gamma enforcement"),
    2  => ("power",                        "(float, default 1.0) used only for inv policy type"),
    3  => ("gamma",                        "(float, default 1.0) momentum of change"),
    4  => ("stepsize",                     "(float, default 1.0) sampling rate on iterations"),
    5  => ("max_lr",                       "(float, default 0.005) max learning rate"),
    6  => ("active_first",                 "(boolean, default True) in alter policy"),
    7  => ("active_period",                "(int64_t, required) in alter policy"),
    8  => ("inactive_period",              "(int64_t, required) in alter policy"),
    9  => ("max_iter",                     "(int, default -1) maximum iterations in this training run"),
    10 => ("num_iter",                     "(int, default 0) number of iterations over which to warmup lr"),
    11 => ("start_multiplier",             "(float, default 0) starting multiplier for learning rate"),
    12 => ("end_multiplier",               "(float, default 0) end multiplier for learning rate"),
    13 => ("multiplier",                   "(float, default 0.5) constant multiplier for learning rate"),
    14 => ("multiplier_1",                 "(float, default 1) start multiplier for learning rate"),
    15 => ("multiplier_2",                 "(float, default 1) end multiplier for learning rate"),
    16 => ("sub_policy_num_iters",         "(int array, default empty) number of iterations for each sub learning rate policy in composite policy"),
    17 => ("m1",                           ""),
    18 => ("n1",                           ""),
    19 => ("m2",                           ""),
    20 => ("n2",                           ""),
    21 => ("m3",                           ""),
    22 => ("start_warmup_multiplier",      "defaults to 0.1"),
    23 => ("constant_warmup_num_iter",     "defaults to 10000000"),
    24 => ("linear_warmup_num_iter",       "defaults to 10000000"),
    25 => ("cyclical_max_lr",              "defaults to 0.05, part of CompositeCyclicalLRPolicy"),
    26 => ("cyclical_step_size",           "defaults to 1000000, part of CompositeCyclicalLRPolicy"),
    27 => ("cyclical_decay",               "defaults to 0.999, part of CompositeCyclicalLRPolicy"),
    28 => ("cosine_min_lr",                "defaults to 0.01, part of CompositeCosineLRPolicy"),
    29 => ("cosine_max_lr",                "defaults to 0.05, part of CompositeCosineLRPolicy"),
    30 => ("cosine_period",                "defaults to 50, part of CompositeCosineLRPolicy"),
    31 => ("cosine_t_mult",                "defaults to 1,0, part of CompositeCosineLRPolicy"),
    32 => ("cosine_lr_shrink",             "defaults to 0.99, part of CompositeCosineLRPolicy")
}

device_inference_function!{
    /* 
    LearningRate, 
     ([](const OperatorDef& def) {
      return std::make_pair(
          std::vector<DeviceOption>{DeviceOption()},
          std::vector<DeviceOption>{def.device_option()});
    }) */ 
}

tensor_inference_function!{LearningRate, /* ([](const OperatorDef&,
                                const vector<TensorShape>& in) {
      vector<TensorShape> out(1);
      out[0] = in[0];
      return out;
    }) */
}

impl<T, Context> LearningRateOp<T,Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator(std::forward<Args>(args)...),
            functor_(nullptr),
            base_lr_(this->template GetSingleArgument<float>("base_lr", FLT_MAX)) 

        CAFFE_ENFORCE_NE(base_lr_, FLT_MAX, "Base learning rate must be set.");
        const string policy =
            this->template GetSingleArgument<string>("policy", "");
        CAFFE_ENFORCE(policy.size(), "Must specify a learning rate policy.");
        functor_.reset(createLearningRateFunctor(policy));
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            int64_t iter =
            OperatorStorage::Input<Tensor>(0, CPU).template data<int64_t>()[0];
        T learning_rate = base_lr_ * (*functor_)(iter);
        // Write to output.
        auto* output = Output(0);
        output->Resize(vector<int64_t>());
        context_.template CopyFromCPU<T>(
            1, &learning_rate, Output(0)->template mutable_data<T>());
        return true;
        */
    }

    pub fn create_learning_rate_functor(policy: &str, arg_prefix: &str) -> *mut dyn LearningRateFunctor<T> {
        todo!();
        /*
        if (policy == "fixed") {
          return new FixedLearningRate<T>();
        } else if (policy == "alter") {
          bool active_first = this->template GetSingleArgument<bool>(
              arg_prefix + "active_first", true);
          int64_t active_period = this->template GetSingleArgument<int64_t>(
              arg_prefix + "active_period", -1);
          int64_t inactive_period = this->template GetSingleArgument<int64_t>(
              arg_prefix + "inactive_period", -1);
          DCHECK_GE(active_period, 0);
          DCHECK_GE(inactive_period, 0);
          return new AlternateLearningRate<T>(
              active_period, inactive_period, active_first);
        } else if (policy == "hill") {
          int64_t num_iter =
              this->template GetSingleArgument<int64_t>(arg_prefix + "num_iter", 0);
          DCHECK_GT(num_iter, 0);
          T start_multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "start_multiplier", 0.);
          DCHECK_GE(start_multiplier, 0); // start_multiplier in range [0, 1]
          DCHECK_LE(start_multiplier, 1);
          T gamma =
              this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
          DCHECK_GT(gamma, 0);
          T power =
              this->template GetSingleArgument<float>(arg_prefix + "power", 0);
          DCHECK_GT(power, 0);
          T end_multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "end_multiplier", 0);
          DCHECK_GE(end_multiplier, 0); // end_multiplier in range [0, 1]
          DCHECK_LE(end_multiplier, 1);
          return new HillLearningRate<T>(
              num_iter, start_multiplier, gamma, power, end_multiplier);
        } else if (policy == "slope") {
          int64_t num_iter_1 = this->template GetSingleArgument<int64_t>(
              arg_prefix + "num_iter_1", 0);
          DCHECK_GT(num_iter_1, 0);
          T multiplier_1 = this->template GetSingleArgument<float>(
              arg_prefix + "multiplier_1", 0.);
          int64_t num_iter_2 = this->template GetSingleArgument<int64_t>(
              arg_prefix + "num_iter_2", 0);
          DCHECK_GT(num_iter_1, 0);
          T multiplier_2 = this->template GetSingleArgument<float>(
              arg_prefix + "multiplier_2", 0.);
          DCHECK_GT(num_iter_2, num_iter_1);
          return new SlopeLearningRate<T>(
              num_iter_1, multiplier_1, num_iter_2, multiplier_2);
        } else if (policy == "step") {
          int stepsize =
              this->template GetSingleArgument<int>(arg_prefix + "stepsize", 0);
          T gamma =
              this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
          DCHECK_GT(stepsize, 0);
          DCHECK_GT(gamma, 0);
          return new StepLearningRate<T>(stepsize, gamma);
        } else if (policy == "exp") {
          T gamma =
              this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
          DCHECK_GT(gamma, 0);
          return new ExpLearningRate<T>(gamma);
        } else if (policy == "gate") {
          T multiplier_1 = this->template GetSingleArgument<float>(
              arg_prefix + "multiplier_1", 1);
          T multiplier_2 = this->template GetSingleArgument<float>(
              arg_prefix + "multiplier_2", 1);
          int num_iter =
              this->template GetSingleArgument<int>(arg_prefix + "num_iter", 0);
          // no constraint on the range of multiplier_1 and multiplier_2
          return new GateLearningRate<T>(multiplier_1, multiplier_2, num_iter);
        } else if (policy == "inv") {
          T gamma =
              this->template GetSingleArgument<float>(arg_prefix + "gamma", 0);
          T power =
              this->template GetSingleArgument<float>(arg_prefix + "power", 0);
          DCHECK_GT(gamma, 0);
          DCHECK_GT(power, 0);
          return new InvLearningRate<T>(gamma, power);
        } else if (policy == "poly") {
          int max_iter =
              this->template GetSingleArgument<int>(arg_prefix + "max_iter", -1);
          T power =
              this->template GetSingleArgument<float>(arg_prefix + "power", 0);
          DCHECK_GT(power, 0);
          return new PolyLearningRate<T>(power, max_iter);
        } else if (policy == "linearWarmup") {
          T start_multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "start_multiplier", 0.);
          int num_iter =
              this->template GetSingleArgument<int>(arg_prefix + "num_iter", 0);
          DCHECK_GE(start_multiplier, 0);
          return new LinearWarmupLearningRate<T>(start_multiplier, num_iter);
        } else if (policy == "constantWarmup") {
          T multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "multiplier", 0.5);
          int num_iter =
              this->template GetSingleArgument<int>(arg_prefix + "num_iter", 0);
          DCHECK_GT(multiplier, 0);
          return new ConstantWarmupLearningRate<T>(multiplier, num_iter);
        } else if (policy == "pieceWarmup") {
          T m1 = this->template GetSingleArgument<float>(arg_prefix + "m1", 0.5);
          int64_t n1 =
              this->template GetSingleArgument<int64_t>(arg_prefix + "n1", 0);
          T m2 = this->template GetSingleArgument<float>(arg_prefix + "m2", 0.5);
          int64_t n2 =
              this->template GetSingleArgument<int64_t>(arg_prefix + "n2", 0);
          T m3 = this->template GetSingleArgument<float>(arg_prefix + "m3", 0.5);
          return new PieceWarmupLearningRate<T>(m1, n1, m2, n2, m3);
        } else if (policy == "composite") {
          std::vector<int> sub_policy_num_iters =
              this->template GetRepeatedArgument<int>("sub_policy_num_iters");
          std::list<CompositeLearningRateItem<T>> sub_policies;
          CAFFE_ENFORCE_GT(
              sub_policy_num_iters.size(),
              0,
              "Must specify at least one sub learning rate policy.");
          for (size_t i = 0; i < sub_policy_num_iters.size(); ++i) {
            CAFFE_ENFORCE_GT(
                sub_policy_num_iters[i],
                0,
                "The number of iterations for sub learning rate policy should be positive.");
            std::stringstream sub_policy_arg_prefix;
            sub_policy_arg_prefix << "sub_policy_" << i << "_";
            const string sub_policy_arg_prefix_str = sub_policy_arg_prefix.str();
            const string sub_policy = this->template GetSingleArgument<string>(
                sub_policy_arg_prefix_str + "policy", "");
            if (sub_policy == "composite") {
              CAFFE_THROW(
                  "Defining composite LR policy as a subpolicy of composite LR "
                  "policy is not allowed.");
            }
            const float scale_lr = this->template GetSingleArgument<float>(
                sub_policy_arg_prefix_str + "lr_scale", 1.0);
            sub_policies.push_back(CompositeLearningRateItem<T>(
                sub_policy_num_iters[i],
                scale_lr,
                createLearningRateFunctor(sub_policy, sub_policy_arg_prefix_str)));
          }
          return new CompositeLearningRate<T>(sub_policies);
        } else if (policy == "cyclical") {
          T max_lr =
              this->template GetSingleArgument<float>(arg_prefix + "max_lr", 0.005);
          int stepsize =
              this->template GetSingleArgument<int>(arg_prefix + "stepsize", 0);
          T decay =
              this->template GetSingleArgument<float>(arg_prefix + "decay", 1.0);
          DCHECK_GT(stepsize, 0);
          DCHECK_GE(max_lr, base_lr_);
          return new CyclicalLearningRate<T>(base_lr_, max_lr, stepsize, decay);
        } else if (policy == "constantThenLinearWarmup") {
          T start_warmup_multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "start_warmup_multiplier", 0.1);
          int64_t constant_warmup_num_iter = this->template GetSingleArgument<int64_t>(
              arg_prefix + "constant_warmup_num_iter", 10000000);
          int64_t linear_warmup_num_iter = this->template GetSingleArgument<int64_t>(
              arg_prefix + "linear_warmup_num_iter", 10000000);
          return new ConstantThenLinearWarmupLearningRate<T>(
              start_warmup_multiplier,
              constant_warmup_num_iter,
              linear_warmup_num_iter);
        } else if (policy == "compositeCyclical") {
          T start_warmup_multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "start_warmup_multiplier", 0.1);
          int64_t constant_warmup_num_iter = this->template GetSingleArgument<int64_t>(
              arg_prefix + "constant_warmup_num_iter", 10000000);
          int64_t linear_warmup_num_iter = this->template GetSingleArgument<int64_t>(
              arg_prefix + "linear_warmup_num_iter", 10000000);
          T cyclical_max_lr = this->template GetSingleArgument<float>(
              arg_prefix + "cyclical_max_lr", 0.05);
          int cyclical_step_size = this->template GetSingleArgument<int>(
              arg_prefix + "cyclical_step_size", 1000000);
          T cyclical_decay = this->template GetSingleArgument<float>(
              arg_prefix + "cyclical_decay", 1.0);
          DCHECK_GE(cyclical_max_lr, base_lr_);
          return new CompositeCyclicalLearningRate<T>(
              base_lr_,
              start_warmup_multiplier,
              constant_warmup_num_iter,
              linear_warmup_num_iter,
              cyclical_max_lr,
              cyclical_step_size,
              cyclical_decay);
        } else if (policy == "cosine") {
          T max_lr =
              this->template GetSingleArgument<float>(arg_prefix + "max_lr", 0.5);
          T min_lr =
              this->template GetSingleArgument<float>(arg_prefix + "min_lr", 0.1);
          int64_t period =
              this->template GetSingleArgument<int>(arg_prefix + "period", 50);
          T t_mult =
              this->template GetSingleArgument<float>(arg_prefix + "t_mult", 1.0);
          T lr_shrink = this->template GetSingleArgument<float>(
              arg_prefix + "lr_shrink", 0.99);
          DCHECK_GE(max_lr, min_lr);
          return new CosineLearningRate<T>(
              min_lr, max_lr, period, t_mult, lr_shrink);
        } else if (policy == "compositeCosine") {
          T start_warmup_multiplier = this->template GetSingleArgument<float>(
              arg_prefix + "start_warmup_multiplier", 0.1);
          int64_t constant_warmup_num_iter = this->template GetSingleArgument<int64_t>(
              arg_prefix + "constant_warmup_num_iter", 10000000);
          int64_t linear_warmup_num_iter = this->template GetSingleArgument<int64_t>(
              arg_prefix + "linear_warmup_num_iter", 10000000);
          T cosine_max_lr = this->template GetSingleArgument<float>(
              arg_prefix + "cosine_max_lr", 0.5);
          T cosine_min_lr = this->template GetSingleArgument<float>(
              arg_prefix + "cosine_min_lr", 0.1);
          int64_t cosine_period = this->template GetSingleArgument<int>(
              arg_prefix + "cosine_period", 50);
          T cosine_t_mult = this->template GetSingleArgument<float>(
              arg_prefix + "cosine_t_mult", 1.0);
          T cosine_lr_shrink = this->template GetSingleArgument<float>(
              arg_prefix + "cosine_lr_shrink", 0.99);

          DCHECK_GE(cosine_max_lr, cosine_min_lr);
          return new CompositeCosineLearningRate<T>(
              start_warmup_multiplier,
              constant_warmup_num_iter,
              linear_warmup_num_iter,
              cosine_min_lr,
              cosine_max_lr,
              cosine_period,
              cosine_t_mult,
              cosine_lr_shrink);
        } else {
          CAFFE_THROW("Unknown learning rate policy: ", policy);
          return NULL;
        }
        */
    }
}
