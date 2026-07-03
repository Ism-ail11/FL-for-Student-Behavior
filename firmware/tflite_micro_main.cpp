#include "model_data.h"

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

namespace {
constexpr int kTensorArenaSize = 196608;
alignas(16) unsigned char tensor_arena[kTensorArenaSize];
}

int main() {
  const tflite::Model* model = tflite::GetModel(student_behavior_model);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }

  tflite::MicroMutableOpResolver<12> resolver;
  resolver.AddConv2D();
  resolver.AddDepthwiseConv2D();
  resolver.AddAdd();
  resolver.AddReshape();
  resolver.AddLogistic();
  resolver.AddMul();
  resolver.AddMaxPool2D();
  resolver.AddMean();
  resolver.AddFullyConnected();
  resolver.AddPad();
  resolver.AddResizeNearestNeighbor();
  resolver.AddSoftmax();

  tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
  if (interpreter.AllocateTensors() != kTfLiteOk) {
    return 2;
  }

  TfLiteTensor* input = interpreter.input(0);
  // Fill input->data.int8 with a preprocessed 320x320x3 frame.

  if (interpreter.Invoke() != kTfLiteOk) {
    return 3;
  }

  TfLiteTensor* output = interpreter.output(0);
  // Decode output tensor: 40 x 40 x 75.
  (void)input;
  (void)output;
  return 0;
}
