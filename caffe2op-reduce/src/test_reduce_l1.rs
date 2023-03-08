crate::ix!();

#[test] fn reduce_l1_example() {

    todo!();

    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceL1",
        ["X"],
        ["Y"],
        axes=(0,1),
        keepdims=0
    )

    workspace.FeedBlob("X", np.random.randint(10, size=(1,2,5,5)).astype(np.float32))
    print("X:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("Y:", workspace.FetchBlob("Y"))

    ```

    **Result**

    ```

    X:
    [[[[ 2.  7.  6.  4.  5.]
       [ 2.  1.  9.  8.  7.]
       [ 4.  9.  1.  0.  0.]
       [ 6.  4.  0.  8.  1.]
       [ 1.  7.  1.  0.  2.]]

      [[ 5.  8.  1.  7.  7.]
       [ 4.  5.  6.  5.  4.]
       [ 1.  9.  6.  6.  3.]
       [ 6.  6.  8.  8.  4.]
       [ 2.  3.  5.  8.  1.]]]]

    Y:
    [[  7.  15.   7.  11.  12.]
     [  6.   6.  15.  13.  11.]
     [  5.  18.   7.   6.   3.]
     [ 12.  10.   8.  16.   5.]
     [  3.  10.   6.   8.   3.]]
    */
}
