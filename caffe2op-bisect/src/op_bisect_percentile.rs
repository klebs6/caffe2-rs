crate::ix!();

use crate::{
    OperatorStorage,
};

/**
 | This operator is to map raw feature values
 | into the percentile representations based on
 | Bisection for more than one feature.
 |
 | The input is the bath of input feature values,
 | with the size of (batch_size, num_feature),
 | where num_feature = F (F >= 1).
 |
 | For each feature, we also need additional
 | information regarding the feature value
 | distribution.
 |
 | There are several vectors to keep data to
 | percentile mappping information as arguments
 | (context):
 |
 | -1. feature raw values (R)
 | -2. feature percentile mapping (P)
 | -3. feature percentile lower bound (L)
 | -4. feature percentile upper bound (U)
 |
 | A toy example:
 | Suppose the sampled data distribution is as follows:
 | 1, 1, 2, 2, 2, 2, 2, 2, 3, 4
 | We have the mapping vectors as follows:
 | R = [1, 2, 3, 4]
 | P = [0.15, 0.55, 0.9, 1.0]
 | L = [0.1, 0.3, 0.9, 1.0]
 | U = [0.2, 0.8, 0.9, 1.0]
 | Where P is computed as (L + U) / 2.
 |
 | For a given list of feature values, X = [x_0,
 | x_1, ..., x_i, ...], for each feature value
 | (x_i) we first apply bisection to find the right
 | index (t), such that R[t] <= x_i < R[t+1].
 |
 | If x_i = R[t], P[t] is returned;
 |
 | otherwise, the interpolation is apply by
 | (R[t], R[t+1]) and (U[t] and L[t]).
 |
 | As there are F features (F >= 1), we concate
 | all the R_f, P_f, L_f, and U_f for each
 | feature f and use an additional input length to
 | keep track of the number of points for each set of
 | raw feature value to percentile mapping.
 |
 | For example, there are two features:
 |
 | R_1 =[0.1, 0.4, 0.5];
 | R_2 = [0.3, 1.2];
 |
 | We will build R = [0.1, 0.4, 0.5, 0.3, 1.2];
 | besides, we have lengths = [3, 2]
 |
 | to indicate the boundaries of the percentile
 | information.
*/
pub struct BisectPercentileOp<Context> {
    //USE_OPERATOR_CONTEXT_FUNCTIONS;
    storage: OperatorStorage,
    context: Context,

    n_features:  i32,
    pct_raw:     Vec<f32>,
    pct_mapping: Vec<f32>,
    pct_lower:   Vec<f32>,
    pct_upper:   Vec<f32>,
    pct_lens:    Vec<i32>,
    index:       Vec<i32>,
    fast_pct:    Vec<HashMap<f32,f32>>,
}

num_inputs!{BisectPercentile, 1}

num_outputs!{BisectPercentile, 1}

inputs!{BisectPercentile, 
    0 => ("raw_values", "Input 2D tensor of floats of size (N, D), where N is the batch size and D is the feature dimension.")
}

outputs!{BisectPercentile, 
    0 => ("percentile", "2D tensor of output with the same dimensions as the input raw_values.")
}

args!{BisectPercentile, 
    0 => ("percentile_raw",     "1D tensor, which is the concatenation of all sorted raw feature values for all features."),
    1 => ("percentile_mapping", "1D tensor. There is one-one mapping between percentile_mapping and percentile_raw such that each element in percentile_mapping corresponds to the percentile value of the corresponding raw feature value."),
    2 => ("percentile_lower",   "1D tensor. There is one-one mapping between percentile_upper and percentile_raw such that each element in percentile_mapping corresponds to the percentile lower bound of the corresponding raw feature value."),
    3 => ("percentile_upper",   "1D tensor. There is one-one mapping between percentile_upper and percentile_raw such that each element in percentile_mapping corresponds to the percentile upper bound of the corresponding raw feature value."),
    4 => ("lengths",            "1D tensor. There is one-one mapping between percentile_upper and percentile_raw such that each element in percentile_mapping corresponds to the percentile upper bound of the corresponding raw feature value.")
}

impl<Context> BisectPercentileOp<Context> {

    pub fn new<Args>(args: Args) -> Self {
        todo!();
        /*
            : Operator<Context>(std::forward<Args>(args)...),
            pct_raw_(OperatorStorage::GetRepeatedArgument<float>(
                "percentile_raw",
                vector<float>{})),
            pct_mapping_(OperatorStorage::GetRepeatedArgument<float>(
                "percentile_mapping",
                vector<float>{})),
            pct_lower_(OperatorStorage::GetRepeatedArgument<float>(
                "percentile_lower",
                vector<float>{})),
            pct_upper_(OperatorStorage::GetRepeatedArgument<float>(
                "percentile_upper",
                vector<float>{})),
            pct_lens_(
                OperatorStorage::GetRepeatedArgument<int>("lengths", vector<int>{})) 

        CAFFE_ENFORCE_EQ(
            pct_raw_.size(),
            pct_mapping_.size(),
            "Feature (raw) data and percentile value dimension should match.");
        CAFFE_ENFORCE_EQ(
            pct_raw_.size(),
            pct_lower_.size(),
            "Feature (raw) data and lower bound dimension should match.");
        CAFFE_ENFORCE_EQ(
            pct_raw_.size(),
            pct_upper_.size(),
            "Feature (raw) data and upper bound dimension should match.");
        n_features = pct_lens_.size();
        index.reserve(n_features + 1);
        index[0] = 0;
        for (int i = 1; i <= n_features; ++i) {
          index[i] = index[i - 1] + pct_lens_[i - 1];
        }
        CAFFE_ENFORCE_EQ(
            index[n_features], // The sum of lengths_data
            pct_raw_.size(),
            "Sum of lengths should be equal to the total number of percentile "
            "mapping data samples");
        */
    }
    
    #[inline] pub fn run_on_device(&mut self) -> bool {
        
        todo!();
        /*
            // Input
        const auto& raw = Input(RAW);
        CAFFE_ENFORCE_EQ(raw.dim(), 2);
        const auto batch_size = raw.size(0);
        const auto num_features = raw.size(1);
        CAFFE_ENFORCE_EQ(num_features, pct_lens_.size());
        const float* raw_data = raw.template data<float>();

        // Output

        auto* pct = Output(PCT, raw.sizes(), at::dtype<float>());
        float* pct_output = pct->template mutable_data<float>();

        // Compute percentile for each raw feature value
        int feature_start_index = 0;
        int feature_length = 0;
        int cur_index = 0;

        for (int i = 0; i < num_features; ++i) {
          cur_index = i;
          feature_start_index = index[i];
          feature_length = pct_lens_[i];
          for (int j = 0; j < batch_size; ++j) {
            pct_output[cur_index] = compute_percentile(
                pct_raw_.begin() + feature_start_index,
                pct_mapping_.begin() + feature_start_index,
                pct_lower_.begin() + feature_start_index,
                pct_upper_.begin() + feature_start_index,
                feature_length,
                raw_data[cur_index]);
            cur_index += num_features;
          }
        }
        return true;
        */
    }

    pub fn binary_search(
        &mut self,
        data: &[f32], 
        lo:   i32, 
        hi:   i32, 
        val:  f32) -> i32 
    {
      todo!();
      /*
        int mid;
        bool low_cond, high_cond;

        while (lo < hi) {
          mid = (lo + hi) >> 1;
          low_cond = (data[mid] <= val);
          high_cond = (val < data[mid + 1]);
          if (low_cond && high_cond) {
            return mid;
          } else if (!low_cond) {
            hi = mid - 1;
          } else {
            lo = mid + 1;
          }
        }
        return lo;
      */
    }

    pub fn compute_percentile(
        &mut self,
        pct_raw_it:     &[f32], 
        pct_mapping_it: &[f32],
        pct_lower_it:   &[f32],
        pct_upper_it:   &[f32],
        size:           i32,
        val:            f32) -> f32 
    {
      todo!();
      /*
        // Corner cases where no interpolation is needed.
        if (val < pct_raw_it[0]) {
          return 0.;
        }
        if (val > pct_raw_it[size - 1]) {
          return 1.;
        }

        float result;
        // Interpolation by binary search
        const auto k = binary_search(pct_raw_it, 0, size - 1, val);

        if (pct_raw_it[k] == val) {
          // Exact match
          result = pct_mapping_it[k];
        } else {
          // interpolation
          float w = (val - pct_raw_it[k]) /
              (pct_raw_it[k + 1] - pct_raw_it[k] + kEPSILON);
          result = (1 - w) * pct_upper_it[k] + w * pct_lower_it[k + 1];
        }
        return result;
      */
    }
}

input_tags!{
    BisectPercentileOp {
        Raw
    }
}

output_tags!{
    BisectPercentileOp {
        Pct
    }
}

pub const BisectPercentileOpEpsilon: f32 = 1e-10;

register_cpu_operator![
    BisectPercentile, 
    BisectPercentileOp<CPUContext>
];

no_gradient!{BisectPercentile}
