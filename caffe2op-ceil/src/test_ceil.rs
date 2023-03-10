crate::ix!();

#[test] fn ceil_op_test() {

    let mut workspace = Workspace::default().reset_workspace();

    let mut op = create_operator(
        "ceil", 
        vec!["X"], 
        vec!["X"], 
        null_mut()
    );

    todo!();
    /*

    workspace.FeedBlob("X", (np.random.uniform(-10, 10, (5,5))).astype(np.float32))
    print("X before running op:", workspace.FetchBlob("X"))
    workspace.RunOperatorOnce(op)
    print("X after running op:", workspace.FetchBlob("X"))

    **Result**

    X before running op:
    [[ 8.44598    -6.5098248  -2.2993476  -7.6859694   0.58566964]
     [-7.846551   -0.03689406  6.9362907  -4.0521703   4.4969673 ]
     [ 0.33355865 -7.895527   -8.393201    9.374202   -2.3930092 ]
     [-6.3061996   3.1403487   3.782099   -8.516556   -2.8387244 ]
     [-2.0164998   4.7663913  -3.422966    0.3636999   8.75713   ]]
    X after running op:
    [[ 9. -6. -2. -7.  1.]
     [-7. -0.  7. -4.  5.]
     [ 1. -7. -8. 10. -2.]
     [-6.  4.  4. -8. -2.]
     [-2.  5. -3.  1.  9.]]
    */
}
