crate::ix!();

#[test] fn reduce_front_mean_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceFrontMean",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[5. 0. 9.]
      [4. 1. 1.]
      [9. 0. 8.]]

     [[2. 6. 7.]
      [6. 2. 6.]
      [0. 4. 5.]]]
    Y: [4.3333335    2.1666667     6.]

    */
}

/**
  | Reduces the input tensor along the last
  | dimension of the by applying **mean**.
  | 
  | Can reduce more than one of the "last"
  | dimensions by setting `num_reduce_dim`.
  | 
  | A second (optional) input, `lengths`,
  | can be passed, which enforces that only
  | a subset of the elements are considered
  | in the mean operation.
  | 
  | - If input tensor `X` has shape $(d_0,
  | d_1, d_2, ..., d_n)$, `lengths` must
  | have shape $(d_0 * d_1 * d_2 * ... * d_{n-1})$.
  | 
  | - The values of the `lengths` tensor
  | determine how many of the values to consider
  | for each vector in the $d_{n-1}$ dimension.
  | 
  | For example if $X = [[1,5,2,9],[4,1,8,2],[2,7,0,3]]$
  | and $lengths = [2,3,1]$, then $Y = [mean(1,5),
  | mean(4,1,8), mean(2)] = [3, 4.333,
  | 2]$
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_front_back_mean_ops.cc
  |
  */
#[test] fn reduce_back_mean_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceBackMean",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[5. 9. 0.]
       [8. 4. 0.]
       [2. 2. 4.]]

      [[9. 0. 9.]
       [7. 9. 7.]
       [1. 0. 2.]]]]
    Y: [[3.7777777 4.888889 ]]

    */
}

/**
  | Computes the **mean** of the input tensor's
  | elements along the provided `axes`.
  |
  | The resulting tensor has the same rank
  | as the input if the `keepdims` argument
  | equals 1 (default). 
  |
  | If `keepdims` is
  | set to 0, then the `axes` dimensions
  | are pruned.
  | 
  | Github Links:
  | 
  | - https://github.com/pytorch/pytorch/blob/master/caffe2/operators/reduce_ops.cc
  |
  */
#[test] fn reduce_mean_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceMean",
        ["X"],
        ["Y"],
        axes=(0,1),
        keepdims=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[9. 0. 3. 6. 0.]
       [3. 4. 5. 0. 9.]
       [6. 9. 1. 1. 5.]
       [6. 2. 3. 7. 7.]
       [3. 1. 1. 0. 1.]]

      [[4. 3. 9. 8. 1.]
       [8. 2. 0. 4. 0.]
       [8. 9. 9. 0. 2.]
       [7. 2. 5. 8. 9.]
       [5. 9. 1. 9. 0.]]]]
    Y:
    [[6.5 1.5 6.  7.  0.5]
     [5.5 3.  2.5 2.  4.5]
     [7.  9.  5.  0.5 3.5]
     [6.5 2.  4.  7.5 8. ]
     [4.  5.  1.  4.5 0.5]]

    */
}
