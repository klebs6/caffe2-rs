crate::ix!();

#[test] fn reduce_front_sum() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceFrontSum",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[4. 1. 1.]
      [0. 6. 7.]
      [7. 8. 6.]]

     [[5. 7. 7.]
      [0. 1. 6.]
      [2. 9. 0.]]]
    Y: [18. 32. 27.]
    */
}

#[test] fn reduce_back_sum_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceBackSum",
        ["X"],
        ["Y"],
        num_reduce_dim=2
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,3,3)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    X:
    [[[[2. 7. 7.]
       [1. 1. 0.]
       [9. 7. 2.]]

      [[6. 6. 4.]
       [1. 2. 6.]
       [6. 6. 3.]]]]
    Y: [[36. 40.]]
    */
}

#[test] fn reduce_sum_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceSum",
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
    [[[[5. 3. 7. 9. 5.]
       [4. 5. 1. 8. 3.]
       [1. 0. 9. 7. 6.]
       [7. 5. 0. 3. 1.]
       [6. 4. 4. 8. 3.]]

      [[8. 9. 6. 7. 7.]
       [5. 5. 4. 7. 0.]
       [9. 7. 6. 6. 7.]
       [7. 5. 2. 4. 2.]
       [4. 5. 1. 9. 4.]]]]
    Y:
    [[13. 12. 13. 16. 12.]
     [ 9. 10.  5. 15.  3.]
     [10.  7. 15. 13. 13.]
     [14. 10.  2.  7.  3.]
     [10.  9.  5. 17.  7.]]

    */
}
