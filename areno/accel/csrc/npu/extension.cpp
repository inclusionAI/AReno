#include <torch/csrc/utils/pybind.h>

void register_activations(pybind11::module_& m);
void register_normalization(pybind11::module_& m);
void register_optimizer(pybind11::module_& m);
void register_embedding(pybind11::module_& m);
void register_linear(pybind11::module_& m);
void register_conv(pybind11::module_& m);

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    register_activations(m);
    register_normalization(m);
    register_optimizer(m);
    register_embedding(m);
    register_linear(m);
    register_conv(m);
    m.attr("activation_implementation") = "ascendc";
    m.attr("normalization_implementation") = "ascendc";
    m.attr("optimizer_implementation") = "ascendc";
    m.attr("embedding_implementation") = "ascendc";
    m.attr("linear_implementation") = "aclnn_ascendc";
    m.attr("conv_implementation") = "ascendc";
    m.attr("supports_training_and_serving") = false;
}
