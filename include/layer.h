#pragma once

#include <vector>
#include "Activation.h"

class Layer{
private:

public:
  Layer(std::vector<std::vector<double>> W, std::vector<double> b);

  double forward(double X);
  double backward(double Y_pred, double Y_true);
};
