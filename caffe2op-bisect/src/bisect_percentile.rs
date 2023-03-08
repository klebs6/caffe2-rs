crate::ix!();

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
#[USE_OPERATOR_CONTEXT_FUNCTIONS]
pub struct BisectPercentileOp<Context> {
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
