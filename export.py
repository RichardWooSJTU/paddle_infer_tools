import paddle
from paddle import _C_ops
from paddle.fluid.framework import _non_static_mode, in_dygraph_mode, _in_legacy_dygraph
from paddle.fluid.layer_helper import LayerHelper

# def op(attn, x, mask):
#     if _non_static_mode():
#         return _C_ops.fused_token_prune(attn, x, mask, 'factor', 0.5)
#     else:
#         helper = LayerHelper('fused_token_prune', **locals())
#         out = helper.create_variable_for_type_inference('float32')
#         helper.append_op(
#             type='fused_token_prune',
#             inputs={'Attn': attn, 'X': x, 'Mask': mask},
#             outputs={'SlimmedX': out},
#             attrs={'factor': 0.5})
#         return out

# op = paddle.jit.to_static(op)
# path = 'output/fused_token_prune'

# paddle.jit.save(
#     op,
#     path,
#     input_spec=[
#         paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32', name='attn'),
#         paddle.static.InputSpec(shape=[None, None, None], dtype='float32', name='x'),
#         paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32', name='mask')],
# )
def take_along_axis(x, indices):
    C = paddle.shape(x)[2]
    # C = 768
    indices = paddle.unsqueeze(indices, axis=-1)
    indices = paddle.tile(indices, repeat_times=(1, 1, C))
    return paddle.take_along_axis(x, indices, axis=1)

def forward(attn, x, mask):
    tensor_shape = paddle.shape(x) #x.shape
    B = tensor_shape[0]#.numpy()[0]
    N = tensor_shape[1]
    C = tensor_shape[2]
    nb_head = paddle.shape(mask)[1]
    attn = attn * paddle.cast(mask >= 0, 'int32')
    attn = paddle.sum(attn, axis = 1)
    attn_by = paddle.sum(attn, axis = 1) #shape is (B, N)

    inds = paddle.argsort(attn_by[:,1:], axis = -1, descending=True) + 1
    cls_ind = paddle.zeros(tensor_shape[0], dtype= paddle.int64 ).unsqueeze(axis=1)
    cls_inds = paddle.concat([cls_ind, paddle.slice(inds, axes=[1], starts=[0], ends=[N-1])], axis=1)
    
    
    max_slimmed_seq_len = int(N * 0.5)
    cls_inds = cls_inds[:,:max_slimmed_seq_len]

    paddle.fluid.layers.Print(x)
    paddle.fluid.layers.Print(cls_inds)
    slimmed_x = take_along_axis(x, cls_inds)
    return slimmed_x

net = paddle.jit.to_static(forward)
path = 'output/net'

paddle.jit.save(
    net,
    path,
    input_spec=[
        paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32', name='attn'),
        paddle.static.InputSpec(shape=[None, None, None], dtype='float32', name='x'),
        paddle.static.InputSpec(shape=[None, 12, None, None], dtype='float32', name='mask')],
)
