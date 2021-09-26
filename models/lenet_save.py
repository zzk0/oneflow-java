from oneflow.compatible import single_client as flow
from lenet_model import lenet


def init_env():
    flow.env.init()
    flow.config.machine_num(1)
    flow.config.cpu_device_num(1)
    flow.config.gpu_device_num(1)
    flow.config.enable_debug_mode(True)


def make_mlp_infer_func():
    input_lbns = {}
    output_lbns = {}

    @flow.global_function(type="predict")
    def mlp_inference(
            images: flow.typing.Numpy.Placeholder((1, 1, 28, 28), dtype=flow.float32)
    ) -> flow.typing.Numpy:
        input_lbns["image"] = images.logical_blob_name
        with flow.scope.placement("gpu", "0:0"):
            logits = lenet(images, train=False)
        output_lbns["output"] = logits.logical_blob_name
        return logits

    return mlp_inference, input_lbns, output_lbns


def save_model():
    init_env()
    mlp_infer, input_lbns, output_lbns = make_mlp_infer_func()
    print(mlp_infer.__name__)
    flow.load_variables(flow.checkpoint.get("./lenet_models_1"))

    saved_model_path = "lenet_models"
    model_name = "lenet"
    model_version = 1

    saved_model_builder = flow.saved_model.ModelBuilder(saved_model_path)
    signature_builder = (
        saved_model_builder.ModelName(model_name)
        .Version(model_version)
        .AddFunction(mlp_infer)
        .AddSignature('lenet')
    )
    for input_name, lbn in input_lbns.items():
        signature_builder.Input(input_name, lbn)
    for output_name, lbn in output_lbns.items():
        signature_builder.Output(output_name, lbn)
    saved_model_builder.Save()

if __name__ == '__main__':
    save_model()
