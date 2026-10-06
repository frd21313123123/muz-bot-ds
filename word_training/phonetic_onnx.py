"""Embed exact CTC whole-word search in ONNX: one scalar probability output."""
import numpy as np


def add_keyword_output(source,destination,temperature):
    import onnx
    from onnx import helper,numpy_helper,TensorProto
    model=onnx.load(str(source));graph=model.graph
    mapping={1:[1,4,4,4,4,5],2:[4,2,4,4,4,5],3:[4,4,3,4,4,5],4:[0,0,0,5,0,5],5:[4,4,4,4,4,5]}
    transition=np.zeros((6,36,36),np.float64)
    for state in range(6):
        for last in range(6):
            for token in range(6):
                following=state if token==0 or token==last else mapping[token][state]
                transition[token,state*6+last,following*6+token]=1
    def tensor(name,value):return numpy_helper.from_array(np.asarray(value),name=name)
    body=helper.make_graph([
        helper.make_node('MatMul',['kw_frame','kw_transition'],['kw_flat']),
        helper.make_node('Reshape',['kw_flat','kw_matrix_shape'],['kw_matrix']),
        helper.make_node('Unsqueeze',['kw_previous','kw_axis1'],['kw_row']),
        helper.make_node('MatMul',['kw_row','kw_matrix'],['kw_next_row']),
        helper.make_node('Squeeze',['kw_next_row','kw_axis1'],['kw_next'])
    ],'whole_word_ctc_step',[
        helper.make_tensor_value_info('kw_previous',TensorProto.DOUBLE,['batch',36]),
        helper.make_tensor_value_info('kw_frame',TensorProto.DOUBLE,['batch',6])
    ],[helper.make_tensor_value_info('kw_next',TensorProto.DOUBLE,['batch',36])],[
        tensor('kw_transition',transition.reshape(6,1296)),tensor('kw_matrix_shape',np.array([-1,36,36],np.int64)),tensor('kw_axis1',np.array([1],np.int64))])
    initial=np.zeros((1,36),np.float64);initial[0,0]=1
    graph.initializer.extend([
        tensor('kw_temperature',np.array(temperature,np.float64)),tensor('kw_initial',initial),
        tensor('kw_zero_index',np.array(0,np.int64)),tensor('kw_axis0',np.array([0],np.int64)),
        tensor('kw_width',np.array([36],np.int64)),tensor('kw_accept',np.array([*range(18,24),*range(30,36)],np.int64)),
        tensor('kw_axis1_outer',np.array([1],np.int64)),tensor('kw_min',np.array(0.,np.float64)),tensor('kw_max',np.array(1.,np.float64))])
    graph.node.extend([
        helper.make_node('Cast',['logits'],['kw_logits'],to=TensorProto.DOUBLE),
        helper.make_node('Div',['kw_logits','kw_temperature'],['kw_scaled']),
        helper.make_node('Softmax',['kw_scaled'],['kw_probability'],axis=2),
        helper.make_node('Shape',['logits'],['kw_input_shape']),
        helper.make_node('Gather',['kw_input_shape','kw_zero_index'],['kw_batch'],axis=0),
        helper.make_node('Unsqueeze',['kw_batch','kw_axis0'],['kw_batch_vector']),
        helper.make_node('Concat',['kw_batch_vector','kw_width'],['kw_state_shape'],axis=0),
        helper.make_node('Expand',['kw_initial','kw_state_shape'],['kw_start']),
        helper.make_node('Scan',['kw_start','kw_probability'],['kw_final'],body=body,num_scan_inputs=1,scan_input_axes=[1]),
        helper.make_node('Gather',['kw_final','kw_accept'],['kw_accepted'],axis=1),
        helper.make_node('ReduceSum',['kw_accepted','kw_axis1_outer'],['kw_sum'],keepdims=0),
        helper.make_node('Clip',['kw_sum','kw_min','kw_max'],['probability'])])
    del graph.output[:]
    graph.output.append(helper.make_tensor_value_info('probability',TensorProto.DOUBLE,['batch']))
    onnx.checker.check_model(model);onnx.save(model,str(destination))
