import time
import argparse
import numpy as np
import paddle
import paddle.inference as paddle_infer

warm_up_times = 10
test_times = 30

bsz = 64
nb_head = 12
max_seq_len = 512
slimmed_seq_len = 128
c = 768

with_trt = True
with_fp16 = False

def infer(predictor, infer_name, attn, x, mask, new_mask):
    input_names = predictor.get_input_names()
    feed_tensors = [predictor.get_input_handle(name) for name in input_names]

    output_names = predictor.get_output_names()
    fetch_tensors = [predictor.get_output_handle(name) for name in output_names]


    for i in range(test_times):
        if i == warm_up_times:
            start = time.time()
        feed_dicts = {
            'attn': attn,
            'x': x,
            'mask': mask,
            'new_mask': new_mask
        }

        for name_i, name in enumerate(input_names):
            feed_tensors[name_i].copy_from_cpu(feed_dicts[name])

        predictor.run()
        out = fetch_tensors[0].copy_to_cpu()
        
    
    infer_predict_time = (time.time() - start) / (test_times - warm_up_times)
    print(infer_name, "推理用时: ", infer_predict_time)
    return out

def check(ndarray1, name1, ndarray2, name2):
    print("{}与{}推理相同？{}".format(name1, name2, 
        np.array_equal(ndarray1, ndarray2)))

def run(path, attn, x, mask, new_mask):
    ######################
    # 原生模型推理#########
    ######################
    loaded_func = paddle.jit.load(path[:-8])

    for i in range(test_times):
        if i == warm_up_times:
            start = time.time()
        attn_t = paddle.to_tensor(attn)
        mask_t = paddle.to_tensor(mask)
        x_t = paddle.to_tensor(x)
        new_mask_t = paddle.to_tensor(new_mask)
        out_origin = loaded_func(attn_t, x_t, mask_t, new_mask_t)
    origin_predict_time = (time.time() - start) / (test_times - warm_up_times)
    out_origin = out_origin.numpy()
    print("原生推理用时: ", origin_predict_time)


    ######################
    #paddle inference 推理#
    ######################
    config = paddle_infer.Config()
    config.set_prog_file(path)
    config.enable_use_gpu(100, 0)
    predictor = paddle_infer.create_predictor(config)
    out_infer = infer(predictor, "paddle inference", attn, x, mask, new_mask)

    
    # print("原生推理与inference推理相同？", np.array_equal(out_origin, out_infer))
    # print("原生推理与trt推理相同？", np.array_equal(out_origin, out_trt))
    # print("inference推理与trt推理相同？", np.array_equal(out_infer, out_trt))
    # diff = out_infer - out_trt
    # print(np.where(diff > 0.000001))
    return out_infer

def run_trt(with_fp16, path, attn, x, mask, new_mask):
    if with_fp16:
        f = 16
        pre_mode = paddle_infer.PrecisionType.Half
    else: 
        f = 32
        pre_mode = paddle_infer.PrecisionType.Float32
    ######################
    #trt 推理#
    ######################
    config = paddle_infer.Config()
    config.set_prog_file(path)
    config.enable_use_gpu(100, 0)
    config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                max_batch_size = 1, 
                                min_subgraph_size = 0, 
                                precision_mode=pre_mode, 
                                use_static = False, use_calib_mode = False)
    config.switch_ir_debug(True)
    min_input_shape = {
        'attn': [1, nb_head, 1, 1],
        'mask': [1, nb_head, 1, 1],
        'x': [1, 1, 1],
        'new_mask': [1, nb_head, 1, 1]
    }
    max_input_shape = {
        'attn': [attn.shape[0], nb_head, attn.shape[2], attn.shape[2]],
        'mask': [attn.shape[0], nb_head, attn.shape[2], attn.shape[2]],
        'x': [x.shape[0], x.shape[1], x.shape[2]],
        'new_mask': [attn.shape[0], nb_head, attn.shape[2], attn.shape[2]]
    }
    opt_input_shape = {
        'attn': [attn.shape[0], nb_head, attn.shape[2], attn.shape[2]],
        'mask': [attn.shape[0], nb_head, attn.shape[2], attn.shape[2]],
        'x': [x.shape[0], x.shape[1], x.shape[2]],
        'new_mask': [new_mask.shape[0], nb_head, new_mask.shape[2], new_mask.shape[2]]
    }
    config.set_trt_dynamic_shape_info(min_input_shape=min_input_shape,
                                  max_input_shape=max_input_shape,
                                  optim_input_shape=opt_input_shape)

    predictor = paddle_infer.create_predictor(config)

    out_trt = infer(predictor, "paddle inference with trt_fp{}".format(f), attn, x, mask, new_mask)

def main():
    args = parse_args()
    dtype = 'float16' if with_fp16 else 'float32'
    print("===================")
    print(dtype)
    attn = np.random.rand(bsz, nb_head, max_seq_len, max_seq_len).astype(dtype)
    mask = np.random.uniform(-1, 1, [bsz, nb_head, max_seq_len, max_seq_len]).astype(dtype)
    x = np.random.rand(bsz, max_seq_len, c).astype(dtype)
    new_mask = np.random.rand(bsz, nb_head, slimmed_seq_len, slimmed_seq_len).astype(dtype)
    
    path = args.first_model_file
    # out = run(path, attn, x, mask, new_mask)
    if with_trt:
        run_trt(with_fp16, path, attn, x, mask, new_mask)
    else:
        run(path, attn, x, mask, new_mask)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--first_model_file", type=str)
    parser.add_argument("--second_model_file", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    main()