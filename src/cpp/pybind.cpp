#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "SwiftTD.h"

namespace py = pybind11;

PYBIND11_MODULE(swift_td, m)
{
     m.doc() = "Python bindings for the SwiftTD reinforcement learning algorithm"; // Module docstring
     py::class_<SwiftTDNonSparse>(m, "SwiftTDNonSparse")
          .def(py::init<int, float, float, float, float, float, float, float, float>(),
               "Initialize the SwiftTDNonSparse algorithm",
               py::arg("num_of_features"),
               py::arg("lambda_"),
               py::arg("alpha"),
               py::arg("gamma"),
               py::arg("epsilon"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("meta_step_size"),
               py::arg("eta_min"))
          .def("step", &SwiftTDNonSparse::Step,
               "Perform one step of learning",
               py::arg("features"),
               py::arg("reward"))
          .def("predict", &SwiftTDNonSparse::Predict,
               "Predict the value without updating model weights",
               py::arg("features"));

     // Bind SwiftTDSparse class
     py::class_<SwiftTDBinaryFeatures>(m, "SwiftTDBinaryFeatures")
          .def(py::init<int, float, float, float, float, float, float, float, float>(),
               "Initialize the SwiftTDBinaryFeatures algorithm",
               py::arg("num_of_features"),
               py::arg("lambda_"),
               py::arg("alpha"),
               py::arg("gamma"),
               py::arg("epsilon"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("meta_step_size"),
               py::arg("eta_min"))
          .def("step", &SwiftTDBinaryFeatures::Step,
               "Perform one step of learning with sparse features",
               py::arg("features_indices"),
               py::arg("reward"))
          .def("predict", &SwiftTDBinaryFeatures::Predict,
               "Predict the value without updating model weights",
               py::arg("features_indices"));

     // Bind SwiftTDSparseAndNonBinaryFeatures class
     py::class_<SwiftTD>(m, "SwiftTD")
          .def(py::init<int, float, float, float, float, float, float, float, float>(),
               "Initialize the SwiftTD algorithm",
               py::arg("num_of_features"),
               py::arg("lambda_"),
               py::arg("alpha"),
               py::arg("gamma"),
               py::arg("epsilon"),
               py::arg("eta"),
               py::arg("decay"),
               py::arg("meta_step_size"),
               py::arg("eta_min"))
          .def("step", &SwiftTD::Step,
               "Perform one step of learning with sparse non-binary features",
               py::arg("feature_indices_values"),
               py::arg("reward"))
          .def("predict", &SwiftTD::Predict,
               "Predict the value without updating model weights",
               py::arg("feature_indices_values"));
}
