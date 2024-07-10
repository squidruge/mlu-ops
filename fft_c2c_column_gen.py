import os
rela_path = './test/mlu_op_gtest/pb_gtest/src/zoo/fft/test_case/'
# rela_path = './test_gen_pb/'
os.system("rm -f " + rela_path + "*")

RADIX3_START = 6000
# RADIX3_END = 10000
# RADIX3_END = 729 * 729*729/9
RADIX3_END = 6000
# RADIX3_END = 9
RADIX3_STRIDE = 1000

RADIX2_START = 256
# RADIX3_END = 10000
# RADIX3_END = 729 * 729*729/9
RADIX2_END = 65536 * 64
# RADIX3_END = 9
RADIX2_STRIDE = 2

radix = 3

batch =64 


if radix == 3:
    radix_start = RADIX3_START
    radix_end = RADIX3_END
    radix_stride = RADIX3_STRIDE

if radix == 2:
    radix_start = RADIX2_START
    radix_end = RADIX2_END
    radix_stride = RADIX2_STRIDE

# for i in range(radix_start, radix_end, radix_stride):
i = radix_start
while i <= radix_end:
    dst = 1 
    stride = batch
    file_path = rela_path + 'fft_' + str(i) + '.prototxt'
    with open(file_path, 'w') as f:
        prototxt_content = "op_name: \"fft\"\n" + \
        "input {\n" + \
        "  id: \"input1\"\n" + \
        "  shape {\n" + \
        "    dims: {0}\n".format(batch) + \
        "    dims: {0}\n".format(i) + \
        "    dim_stride: {0}\n".format(dst) + \
        "    dim_stride: {0}\n".format(stride) + \
        "  }\n" + \
        "  layout: LAYOUT_ARRAY\n" + \
        "  dtype: DTYPE_COMPLEX_FLOAT\n" + \
        "  random_data {\n" + \
        "    seed: 23\n" + \
        "    distribution: UNIFORM\n" + \
        "    lower_bound_double: -10\n" + \
        "    upper_bound_double: 10\n" + \
        "  }\n" + \
        "  onchip_dtype: DTYPE_FLOAT\n" + \
        "}\n" + \
        "output {\n" + \
        "  id: \"output1\"\n" + \
        "  shape {\n" + \
        "    dims: {0}\n".format(batch) + \
        "    dims: {0}\n".format(i) + \
        "    dim_stride: {0}\n".format(dst) + \
        "    dim_stride: {0}\n".format(stride) + \
        "  }\n" + \
        "  layout: LAYOUT_ARRAY\n" + \
        "  dtype: DTYPE_COMPLEX_FLOAT\n" + \
        "  thresholds {\n" + \
        "    evaluation_threshold: 1e-05\n" + \
        "    evaluation_threshold: 1e-05\n" + \
        "    evaluation_threshold_imag: 1e-05\n" + \
        "    evaluation_threshold_imag: 1e-05\n" + \
        "  }\n" + \
        "}\n" + \
        "evaluation_criterion: DIFF1\n" + \
        "evaluation_criterion: DIFF2\n" + \
        "supported_mlu_platform: MLU370\n" + \
        "handle_param {\n" + \
        "  round_mode: ROUND_OFF_ZERO\n" + \
        "}\n" + \
        "fft_param {\n" + \
        "  rank: 1\n" + \
        "  n: {0}\n".format(i) + \
        "  direction: 1\n" + \
        "  scale_factor: 1\n" + \
        "}\n" + \
        "test_param: {\n" + \
        "  error_func: DIFF1\n" + \
        "  error_func: DIFF2\n" + \
        "  error_threshold: 1e-05\n" + \
        "  error_threshold: 1e-05\n" + \
        "  baseline_device: CPU\n" + \
        "}"
        f.write(prototxt_content)
        i *= radix_stride
    



