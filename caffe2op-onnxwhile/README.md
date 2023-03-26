# ONNXWhileOp

The `ONNXWhileOp` is a Rust crate defining
a mathematical operator used in digital signal
processing (DSP) and machine learning
computations. This crate provides an
implementation of a loop operator, as specified in
the Open Neural Network Exchange (ONNX) standard.

**Note: This crate is currently being translated from C++ to Rust, and some function bodies may still be in the process of translation.**

The `ONNXWhileOp` allows a graph of operators to
be iterated over multiple times, using a set of
loop-carried dependencies (LCD) tensors. These
tensors are updated in each iteration of the loop,
and passed back into the loop for the next
iteration. This creates a powerful tool for
building complex models, such as recurrent neural
networks and dynamic graphs.

The `ONNXWhileOp` also provides several
convenience functions for working with LCD
tensors, including `set_iteration` and
`OnnxWhileOpLocalScope`. Additionally, the crate
provides a `workspace` abstraction for managing
the data used by the loop, and an `allow_inplace`
flag for indicating when in-place updates are
allowed.

The mathematics behind the `ONNXWhileOp` operator
can be expressed using standard notation for
iterative computations. Given an initial set of
LCD tensors `lcd_tensor`, and a graph of operators
`net`, the loop updates the LCD tensors and
executes the graph in each iteration. This can be
represented as follows:

```
for i in 1..iteration {
    set_iteration(i)
    lcd_tensor = do_run_with_type(&net, &workspace, &lcd_tensor, &allow_inplace)
}
```

Here, `do_run_with_type` is a function that
executes the graph of operators `net` on the input
tensors `lcd_tensor`, using the `workspace` to
manage the data, and respecting the
`allow_inplace` flag. The `set_iteration` function
sets the current iteration number, and is used by
other operators to determine their behavior.

Overall, the `ONNXWhileOp` crate provides
a powerful tool for building iterative models in
Rust, using a standardized format that can be
easily integrated with other ONNX-compliant tools.

2> what are the use cases for running a network in
a while loop?

There are several use cases where running
a network in a while loop can be useful:

1. Recurrent Neural Networks (RNNs): RNNs are
   a type of neural network that are designed to
   process sequential data, such as time-series or
   natural language. RNNs typically use a loop
   structure to process each element of the
   sequence, and the `ONNXWhileOp` can be used to
   implement this loop.

2. Conditional Computation: In some cases, it may
   be necessary to run a network for a variable
   number of iterations, depending on some
   condition. For example, in image segmentation
   tasks, the number of objects in an image may
   vary, and the `ONNXWhileOp` can be used to
   segment each object in the image, with the
   number of iterations determined by the number
   of objects.

3. Reinforcement Learning: In reinforcement
   learning, an agent interacts with an
   environment and takes actions to maximize
   a reward signal. The environment can be modeled
   as a loop, where the agent takes actions and
   receives rewards in each iteration. The
   `ONNXWhileOp` can be used to implement this
   loop, allowing the agent to learn and adapt
   over time.

Overall, the `ONNXWhileOp` provides a flexible and
powerful way to implement iterative computations
in Rust, and can be used in a wide range of
applications across DSP and machine learning.

3> could you show me an example of each of these?

Sure! Here are some examples of how the
`ONNXWhileOp` can be used in each of the use cases
I mentioned earlier:

## Recurrent Neural Networks (RNNs)

One common use case for the `ONNXWhileOp` is to
implement a recurrent neural network (RNN) in
Rust. RNNs are a type of neural network that are
designed to process sequential data, such as
time-series or natural language. In an RNN, the
same set of weights is applied to each element of
the sequence, with the output from one step being
fed back as input to the next step.

Here's an example of how the `ONNXWhileOp` can be
used to implement a simple RNN in Rust, using the
`Gluon` machine learning library:

```rust
use gluon::vm::api::{Hole, OpaqueValue, Userdata};
use gluon::{vm, Thread};

use onnxwhileop::{ONNXWhileOp, OnnxWhileOpLocalScope, run_on_device, workspace};

// Define the RNN graph
let mut net = ONNXWhileOp::new(vec![], vec![], vec![], vec![], vec![]);
net.append_op("linear", vec!["input", "weight", "bias"], vec!["output"]);
net.append_op("tanh", vec!["output"], vec!["output"]);

// Define the input and output tensors
let input = workspace()
    .new_tensor("input", &[1, 1, 10])
    .unwrap()
    .into();

let output = workspace()
    .new_tensor("output", &[1, 1, 10])
    .unwrap()
    .into();

// Run the RNN for 5 iterations
let iteration = 5;

for i in 1..=iteration {
    let iteration_hole = Hole::new::<i64>();
    let mut scope = OnnxWhileOpLocalScope::new();
    scope.add("input", input.clone());
    scope.add("weight", output.clone());
    scope.add("bias", output.clone());
    scope.add("output", output.clone());
    scope.add("iteration_number", iteration_hole.clone().into());

    let (ret, _, _) = run_on_device(
        &mut net,
        &workspace(),
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &scope,
        &iteration_hole,
    )
    .unwrap();
}
```

This example defines a simple RNN graph with
a single linear layer followed by a hyperbolic
tangent activation function. The `ONNXWhileOp` is
used to run the graph for 5 iterations, with the
output from each iteration being fed back as input
to the next iteration.

## Conditional Computation

Another use case for the `ONNXWhileOp` is to
implement conditional computation, where the
number of iterations is determined by some
condition. For example, in image segmentation
tasks, the number of objects in an image may vary,
and the `ONNXWhileOp` can be used to segment each
object in the image, with the number of iterations
determined by the number of objects.

Here's an example of how the `ONNXWhileOp` can be
used to implement conditional computation in Rust,
using the `Gluon` machine learning library:

```rust
use gluon::vm::api::{Hole, OpaqueValue, Userdata};
use gluon::{vm, Thread};

use onnxwhileop::{ONNXWhileOp, OnnxWhileOpLocalScope, run_on_device, workspace};

// Define the object segmentation graph
let mut net = ONNXWhileOp::new(vec!

4> continue

let mut net = ONNXWhileOp::new(vec![], vec![], vec![], vec![], vec![]);
net.append_op("image_segmentation", vec!["image"], vec!["mask"]);
net.append_op("count_objects", vec!["mask"], vec!["num_objects"]);

// Define the input and output tensors
let image = workspace()
    .new_tensor("image", &[1, 3, 256, 256])
    .unwrap()
    .into();

let mask = workspace()
    .new_tensor("mask", &[1, 1, 256, 256])
    .unwrap()
    .into();

// Run the object segmentation graph until all objects have been segmented
let mut num_objects = 0;
let mut i = 0;

while num_objects < max_objects {
    let iteration_hole = Hole::new::<i64>();
    let mut scope = OnnxWhileOpLocalScope::new();
    scope.add("image", image.clone());
    scope.add("mask", mask.clone());
    scope.add("iteration_number", iteration_hole.clone().into());

    let (ret, _, _) = run_on_device(
        &mut net,
        &workspace(),
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &scope,
        &iteration_hole,
    )
    .unwrap();

    num_objects = ret[0].as_ref().unwrap().to_scalar::<f32>().unwrap() as usize;
    i += 1;
}
```

This example defines an object segmentation graph
that takes an input image and produces a binary
mask indicating which pixels belong to which
object. The `ONNXWhileOp` is used to run the graph
until a maximum number of objects have been
segmented, with the number of iterations
determined dynamically by the number of objects.

## Custom Loops

Finally, the `ONNXWhileOp` can be used to
implement custom loops that are not easily
expressible using existing machine learning
operations. For example, the `ONNXWhileOp` could
be used to implement a custom optimization
algorithm that requires iterating over a complex
set of equations.

Here's an example of how the `ONNXWhileOp` can be
used to implement a custom loop in Rust:

```rust
use onnxwhileop::{ONNXWhileOp, OnnxWhileOpLocalScope, run_on_device, workspace};

// Define the loop graph
let mut net = ONNXWhileOp::new(vec![], vec![], vec![], vec![], vec![]);
net.append_op("update_x", vec!["x", "v"], vec!["x"]);
net.append_op("update_v", vec!["x", "v"], vec!["v"]);

// Define the input and output tensors
let x = workspace().new_tensor("x", &[1]).unwrap().into();
let v = workspace().new_tensor("v", &[1]).unwrap().into();

// Define the loop condition
let mut condition = String::from("");
condition.push_str("cond = (i < max_iter)");

// Define the loop body
let mut body = String::from("");
body.push_str("v = v + lr * grad");
body.push_str("x = x - v");

// Run the custom loop
let max_iter = 10;
let lr = 0.1;
let grad = 0.5;

let iteration_hole = Hole::new::<i64>();
let mut scope = OnnxWhileOpLocalScope::new();
scope.add("x", x.clone());
scope.add("v", v.clone());
scope.add("i", 0i64.into());
scope.add("max_iter", max_iter.into());


5> continue

let mut i = 0i64;

while run_on_device(
    &mut net,
    &workspace(),
    &vec![],
    &vec![],
    &vec![],
    &vec![],
    &vec![],
    &vec![],
    &scope,
    &iteration_hole,
)
.is_ok()
{
    if !condition.eval(&scope).unwrap().to_scalar::<bool>().unwrap() {
        break;
    }

    scope.add("i", (i + 1).into());

    let (ret, _, _) = run_on_device(
        &mut net,
        &workspace(),
        &vec!["grad"],
        &vec![grad.clone()],
        &vec![],
        &vec![],
        &vec![],
        &vec![],
        &scope,
        &iteration_hole,
    )
    .unwrap();

    i += 1;
}
```

In this example, the `ONNXWhileOp` is used to
implement a custom optimization algorithm that
updates a variable `x` according to the gradient
of a loss function. The loop has a fixed number of
iterations determined by the `max_iter` parameter,
and the learning rate is controlled by the `lr`
parameter. The loop condition is defined by the
string `cond = (i < max_iter)`, and the loop body
is defined by the string `v = v + lr * grad;
x = x - v`. Inside the loop, the gradient is
updated by the value of the `grad` parameter, and
the iteration number is updated by adding 1 to the
previous value of `i`.

Overall, the `ONNXWhileOp` provides a powerful and
flexible way to implement custom loops in Rust
that can be integrated with existing machine
learning frameworks. By defining custom loop
conditions and bodies, developers can implement
a wide range of algorithms that are not easily
expressible using existing machine learning
operations.

6> what about an example of reinforcement
learning?

Sure, here's an example of using the `ONNXWhileOp`
to implement a simple reinforcement learning
algorithm:

```rust
use onnxruntime::{environment::Environment, tensor::Tensor};
use onnxruntime::{Graph, LoggingLevel};
use rand::{distributions::Uniform, prelude::*};
use std::collections::HashMap;

fn main() {
    // Create the ONNX runtime environment
    let env = Environment::new(LoggingLevel::Warning).unwrap();

    // Load the ONNX model defining the reinforcement learning algorithm
    let mut graph = Graph::new();
    graph.import("reinforcement_learning.onnx").unwrap();

    // Initialize the state and reward variables
    let mut state = Tensor::from(vec![0f32]);
    let mut reward = Tensor::from(vec![0f32]);

    // Set up the initial action distribution
    let actions = vec!["move left", "move right", "stay"];
    let mut action_probs = HashMap::new();
    action_probs.insert("move left", 0.33);
    action_probs.insert("move right", 0.33);
    action_probs.insert("stay", 0.34);

    // Set up the loop variables
    let mut iteration = 0;
    let max_iterations = 100;

    // Define the loop body
    let mut body = String::new();
    body.push_str("state, reward, action_probs = f(state, reward, action_probs); ");
    body.push_str("action = sample_action(action_probs); ");
    body.push_str("state, reward = execute_action(action); ");

    // Define the loop condition
    let condition = "iteration < max_iterations";

    // Define the loop scope
    let mut scope = HashMap::new();
    scope.insert("state".to_string(), state.clone());
    scope.insert("reward".to_string(), reward.clone());
    scope.insert("action_probs".to_string(), Tensor::from(action_probs.values().cloned().collect::<Vec<f32>>()));
    scope.insert("max_iterations".to_string(), Tensor::from(max_iterations));
    scope.insert("iteration".to_string(), Tensor::from(iteration));
    scope.insert("f".to_string(), graph.get_function("f").unwrap());
    scope.insert("sample_action".to_string(), graph.get_function("sample_action").unwrap());
    scope.insert("execute_action".to_string(), graph.get_function("execute_action").unwrap());

    // Run the loop
    while onnx_while_op::run_while_op::<HashMap<String, Tensor>>(
        &mut graph,
        &env,
        &mut scope,
        &condition,
        &body,
    )
    .is_ok()
    {
        // Update the iteration counter
        iteration += 1;

        // Get the updated state and reward
        state = scope.get("state").unwrap().clone();
        reward = scope.get("reward").unwrap().clone();
    }

    // Print the final state and reward
    println!("Final state: {:?}", state);
    println!("Final reward: {:?}", reward);
}
```

In this example, the `ONNXWhileOp` is used to
implement a simple reinforcement learning
algorithm that learns to navigate a 1D
environment. The loop body calls an ONNX function
`f` that takes the current state, reward, and
action probabilities as inputs, and returns
updated values for the state, reward, and action
probabilities. The body also samples an action
from the current action probabilities, and
executes the action to obtain a new state and
reward. The loop condition is based on a fixed
number of iterations, and the loop scope contains
the loop variables and ONNX functions used in the
loop body.

Overall, the `ONNXWhileOp` provides a powerful and

flexible way to define and run loops in ONNX
models, which is useful for a variety of machine
learning and data processing tasks. It allows
users to define complex loop bodies and conditions
using ONNX functions, and provides efficient
memory management and parallel execution for
large-scale datasets. By leveraging the power of
the ONNX format and the ONNX runtime, users can
build and deploy high-performance loop-based
algorithms in a variety of environments, from edge
devices to cloud clusters.

Some other examples of machine learning tasks that
can be implemented using the `ONNXWhileOp`
include:

- Gradient descent: In a typical neural network
  training pipeline, the weights of the network
  are updated iteratively using a gradient descent
  algorithm. The `ONNXWhileOp` can be used to
  implement this algorithm by defining a loop body
  that computes the gradients of the loss function
  with respect to the weights, and updates the
  weights accordingly.

- Attention mechanisms: Attention mechanisms are
  widely used in natural language processing and
  other applications to selectively focus on
  different parts of the input. The `ONNXWhileOp`
  can be used to implement attention mechanisms by
  defining a loop body that iteratively computes
  attention weights based on the current input and
  context, and updates the context based on the
  attention weights.

- Data augmentation: Data augmentation is a common
  technique used in machine learning to increase
  the size and diversity of training datasets. The
  `ONNXWhileOp` can be used to implement data
  augmentation algorithms by defining a loop body
  that applies a series of random transformations
  to the input data, and yields the augmented data
  as output.

Overall, the `ONNXWhileOp` provides a powerful and
flexible way to implement complex iterative
algorithms in ONNX models, and is a valuable tool
for machine learning researchers and
practitioners.

8> could you show me an example of each of these?
(gradient descent, attention mechanisms, data
augmentation)?

Sure, here are some examples of how the
`ONNXWhileOp` can be used to implement gradient
descent, attention mechanisms, and data
augmentation in ONNX models:

### Gradient Descent
Gradient descent is a popular optimization
algorithm used in machine learning to train neural
networks. Here's an example of how the
`ONNXWhileOp` can be used to implement gradient
descent:

```python
import onnx
import numpy as np
from onnx import numpy_helper
from onnx import helper
from onnxruntime import InferenceSession, SessionOptions

# Define the inputs to the model
x = np.array([[1, 2], [3, 4]], dtype=np.float32)
y = np.array([[0], [1]], dtype=np.float32)

# Define the model graph
graph = helper.GraphProto()
graph.name = 'gradient_descent'
input_x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [2, 2])
input_y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 1])
output_loss = helper.make_tensor_value_info('loss', onnx.TensorProto.FLOAT, [])
output_w = helper.make_tensor_value_info('w', onnx.TensorProto.FLOAT, [2, 1])
w = np.array([[0], [0]], dtype=np.float32)
init_w = helper.make_tensor('init_w', onnx.TensorProto.FLOAT, [2, 1], w.flatten())
step_size = np.array([0.1], dtype=np.float32)
init_step_size = helper.make_tensor('init_step_size', onnx.TensorProto.FLOAT, [1], step_size)
graph.initializer.extend([init_w, init_step_size])
loss_node = helper.make_node('ReduceSumSquare', ['y', 'w'], ['loss'], keepdims=0)
diff_node = helper.make_node('Sub', ['y', 'w'], ['diff'])
grad_node = helper.make_node('MatMul', ['x', 'diff'], ['grad'])
update_node = helper.make_node('Mul', ['grad', 'step_size'], ['update'])
new_w_node = helper.make_node('Add', ['w', 'update'], ['new_w'])
cond_node = helper.make_node('Less', ['loss', helper.make_tensor('', onnx.TensorProto.FLOAT, [], [0.001])], ['cond'])
body = helper.make_graph([loss_node, diff_node, grad_node, update_node, new_w_node, cond_node], 'gradient_descent_body', [input_x, input_y], [output_loss, output_w])
loop_inputs = [init_w, init_step_size]
loop_outputs = ['w', 'step_size']
loop_node = helper.make_node('Loop', ['x'], loop_outputs, body, loop_inputs)
graph.node.extend([loop_node])
graph.output.extend([helper.make_tensor_value_info('final_w', onnx.TensorProto.FLOAT, [2, 1])])

# Save the model graph to a file
onnx.save(graph, 'gradient_descent.onnx')

# Run the model with an ONNX runtime session
session_options = SessionOptions()
session_options.optimized_model_filepath = 'gradient_descent_optimized.onnx'
session = InferenceSession('gradient_descent.onnx', session_options)
input_feed = {'x': x, 'y': y}
output = session.run(None, input_feed)
final_w = numpy_helper.to_array(output[0])
print('Final weights:', final_w)
```

In this example, the `ONNXWhileOp` is used to
implement a simple gradient descent algorithm to
find the optimal weights `w` for a linear
regression problem. The algorithm iteratively
computes the gradient of the loss function with
respect to

9> continue

the weights, updates the weights by subtracting
the gradient multiplied by a step size, and
repeats until the loss function falls below
a certain threshold.

### Attention Mechanisms
Attention mechanisms are a key component of many
modern neural network architectures, especially
those used for natural language processing and
image classification. Here's an example of how the
`ONNXWhileOp` can be used to implement an
attention mechanism:

```python
import onnx
import numpy as np
from onnx import helper
from onnxruntime import InferenceSession, SessionOptions

# Define the inputs to the model
x = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=np.float32)
h = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32)

# Define the model graph
graph = helper.GraphProto()
graph.name = 'attention_mechanism'
input_x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [3, 3])
input_h = helper.make_tensor_value_info('h', onnx.TensorProto.FLOAT, [2, 3])
output_y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [2, 3])
output_alpha = helper.make_tensor_value_info('alpha', onnx.TensorProto.FLOAT, [2, 3])
y0 = helper.make_node('MatMul', ['h', helper.make_tensor('', onnx.TensorProto.FLOAT, [3, 3], np.ones((3, 3)).flatten())], ['y0'])
alpha0 = helper.make_node('MatMul', ['h', helper.make_tensor('', onnx.TensorProto.FLOAT, [3, 1], np.ones((3, 1)).flatten())], ['alpha0'])
concat_input = helper.make_node('Concat', ['h', 'y0', alpha0], ['concat_input'], axis=1)
concat_output = helper.make_node('MatMul', ['concat_input', helper.make_tensor('', onnx.TensorProto.FLOAT, [7, 7], np.ones((7, 7)).flatten())], ['concat_output'])
y = helper.make_node('Slice', ['concat_output'], ['y'], starts=[0, 0], ends=[2, 3])
alpha = helper.make_node('Slice', ['concat_output'], ['alpha'], starts=[2, 0], ends=[4, 3])
cond_node = helper.make_node('Less', ['alpha', helper.make_tensor('', onnx.TensorProto.FLOAT, [], [0.9])], ['cond'])
body = helper.make_graph([y, alpha, cond_node], 'attention_mechanism_body', [input_x, input_h], [output_y, output_alpha])
loop_inputs = [helper.make_tensor('', onnx.TensorProto.FLOAT, [2, 3], np.zeros((2, 3)).flatten())]
loop_outputs = ['y', 'alpha']
loop_node = helper.make_node('Loop', ['x'], loop_outputs, body, loop_inputs)
graph.node.extend([loop_node])
graph.output.extend([output_y, output_alpha])

# Save the model graph to a file
onnx.save(graph, 'attention_mechanism.onnx')

# Run the model with an ONNX runtime session
sessionh_options = SessionOptions()
session_options.optimized_model_filepath = 'attention_mechanism_optimized.onnx'
session = InferenceSession('attention_mechanism.onnx', session_options)
input_feed = {'x': x, 'h': h}
output = session.run(None, input_feed)
y = output[0]
```

10> continue

### Data Augmentation
Data augmentation is a technique used to increase
the amount and diversity of data available for
training a machine learning model. By applying
various transformations to the training data, such
as rotations, translations, and scaling, the model
can learn to be more robust to variations in the
input data. Here's an example of how the
`ONNXWhileOp` can be used to implement data
augmentation:

```python
import onnx
import numpy as np
from onnx import helper
from onnxruntime import InferenceSession, SessionOptions

# Define the inputs to the model
x = np.array([[[[1, 2, 3], [4, 5, 6], [7, 8, 9]]]], dtype=np.float32)

# Define the model graph
graph = helper.GraphProto()
graph.name = 'data_augmentation'
input_x = helper.make_tensor_value_info('x', onnx.TensorProto.FLOAT, [1, 1, 3, 3])
output_y = helper.make_tensor_value_info('y', onnx.TensorProto.FLOAT, [8, 1, 3, 3])
y0 = helper.make_node('Identity', ['x'], ['y0'])
cond_node = helper.make_node('Less', ['i', helper.make_tensor('', onnx.TensorProto.INT64, [], [8])], ['cond'])
flip_node = helper.make_node('Flip', ['y', helper.make_tensor('', onnx.TensorProto.INT64, [], [2])], ['flip_output'])
rot_node = helper.make_node('Rotate', ['y', helper.make_tensor('', onnx.TensorProto.FLOAT, [], [30])], ['rot_output'])
crop_node = helper.make_node('Crop', ['y', 'crop'], ['crop_output'])
pad_node = helper.make_node('Pad', ['y', 'pad'], ['pad_output'])
resize_node = helper.make_node('Resize', ['y', 'size'], ['resize_output'])
add_node = helper.make_node('Add', ['y', helper.make_tensor('', onnx.TensorProto.FLOAT, [1, 1, 1, 1], [1])], ['add_output'])
loop_body = helper.make_node('If', ['cond'], ['loop_body_output'], [flip_node, rot_node, crop_node, pad_node, resize_node, add_node])
loop_inputs = [helper.make_tensor('', onnx.TensorProto.INT64, [], [0])]
loop_outputs = ['y']
loop_node = helper.make_node('Loop', ['i'], loop_outputs, loop_body, loop_inputs)
output_y0 = helper.make_node('Identity', ['y'], ['output_y0'])
output_node = helper.make_node('Concat', ['output_y0', 'y'], ['y_concat'], axis=0)
graph.node.extend([y0, cond_node, loop_node, output_y0, output_node])
graph.output.extend([output_y])

# Save the model graph to a file
onnx.save(graph, 'data_augmentation.onnx')

# Run the model with an ONNX runtime session
session_options = SessionOptions()
session_options.optimized_model_filepath = 'data_augmentation_optimized.onnx'
session = InferenceSession('data_augmentation.onnx', session_options)
input_feed = {'x': x, 'i': np.array(0, dtype=np.int64), 'crop': np.array([0, 0], dtype=np.int64), 'pad': np.array([0, 0, 0, 0], dtype=np.int64), 'size': np.array([5, 5], dtype=np.int64)}
output = session.run(None, input_feed)
y = output[0]
```
