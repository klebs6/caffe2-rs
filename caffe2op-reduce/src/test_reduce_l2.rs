crate::ix!();

#[test] fn reduce_l2_example() {

    todo!();
    /*
    workspace.ResetWorkspace()

    op = core.CreateOperator(
        "ReduceL2",
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
    [[[[ 8.  0.  2.  5.  1.]
       [ 1.  3.  0.  4.  0.]
       [ 1.  3.  6.  7.  7.]
       [ 6.  9.  8.  4.  6.]
       [ 6.  1.  5.  7.  3.]]

      [[ 2.  4.  6.  2.  8.]
       [ 1.  1.  8.  0.  8.]
       [ 5.  9.  0.  3.  2.]
       [ 1.  7.  3.  7.  3.]
       [ 6.  8.  9.  8.  7.]]]]

    Y:
    [[  8.24621105   4.           6.3245554    5.38516474   8.06225777]
     [  1.41421354   3.1622777    8.           4.           8.        ]
     [  5.09901953   9.48683262   6.           7.6157732    7.28010988]
     [  6.08276272  11.40175438   8.54400349   8.06225777   6.70820379]
     [  8.48528099   8.06225777  10.29563046  10.63014603   7.6157732 ]]

    */
}
