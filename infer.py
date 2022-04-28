import time
import argparse
import numpy as np
import paddle
import paddle.inference as paddle_infer

warm_up_times = 20
test_times = 120

def infer(predictor, infer_name, attn, x, mask):
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
        }

        for name_i, name in enumerate(input_names):
            feed_tensors[name_i].copy_from_cpu(feed_dicts[name])

        predictor.run()
        # out = fetch_tensors[0].copy_to_cpu()

    infer_predict_time = (time.time() - start) / (test_times - warm_up_times)
    print(infer_name, "推理用时: ", infer_predict_time)

def main():
    attn = np.random.rand(16, 12, 128, 128).astype('float32')
    mask = np.random.uniform(-1, 1, [16, 12, 128, 128]).astype('float32')
    x = np.random.rand(16, 128, 768).astype('float32')



    args = parse_args()
    path = args.model_file
    print(path)
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
        loaded_func(attn_t, x_t, mask_t)
    origin_predict_time = (time.time() - start) / (test_times - warm_up_times)
    print("原生推理用时: ", origin_predict_time)


    ######################
    #paddle inference 推理#
    ######################
    config = paddle_infer.Config()
    config.set_prog_file(path)
    config.enable_use_gpu(100, 0)
    predictor = paddle_infer.create_predictor(config)
    infer(predictor, "paddle inference", attn, x, mask)



    ######################
    #trt 推理#
    ######################
    config = paddle_infer.Config()
    config.set_prog_file(path)
    config.enable_use_gpu(100, 0)
    config.enable_tensorrt_engine(workspace_size = 1 << 30, 
                                max_batch_size = 1, 
                                min_subgraph_size = 0, 
                                precision_mode=paddle_infer.PrecisionType.Float32, 
                                use_static = False, use_calib_mode = False)
    config.switch_ir_debug(True)
    min_input_shape = {
        'attn': [1, 12, 1, 1],
        'mask': [1, 12, 1, 1],
        'x': [1, 1, 1]
    }
    max_input_shape = {
        'attn': [attn.shape[0], 12, attn.shape[2], attn.shape[2]],
        'mask': [attn.shape[0], 12, attn.shape[2], attn.shape[2]],
        'x': [x.shape[0], x.shape[1], x.shape[2]]
    }
    opt_input_shape = {
        'attn': [attn.shape[0], 12, attn.shape[2], attn.shape[2]],
        'mask': [attn.shape[0], 12, attn.shape[2], attn.shape[2]],
        'x': [x.shape[0], x.shape[1], x.shape[2]]
    }
    config.set_trt_dynamic_shape_info(min_input_shape=min_input_shape,
                                  max_input_shape=max_input_shape,
                                  optim_input_shape=opt_input_shape)

    predictor = paddle_infer.create_predictor(config)
    infer(predictor, "paddle inference with trt_fp32", attn, x, mask)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_file", type=str)
    return parser.parse_args()

if __name__ == "__main__":
    main()