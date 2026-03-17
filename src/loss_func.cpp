#include <vector>
#include <cmath>

double mse(const std::vector<double>& Y_true, const std::vector<double>& Y_pred){
  double loss = 0.0;
  for(int i=0; i<Y_pred.size(); i++){
    loss += pow(Y_true[i] - Y_pred[i], 2);
  }
  loss /= Y_pred.size();
  return loss;
}
std::vector<double> mse_derivative(const std::vector<double>& Y_true, const std::vector<double>& Y_pred){ 
  // Upcoming Feature: mini-batch, d_mse(B, output)
  std::vector<double> d_mse(Y_pred.size(), 0.0);
  for(int i=0; i<Y_pred.size(); i++){
    d_mse[i] = (2.0/Y_pred.size())*(Y_pred[i] - Y_true[i]);
  } 
  return d_mse;
}

double bce(const std::vector<double>& Y_true, const std::vector<double>& Y_pred){
  double loss = 0.0;
  double eps = 0.0000001;
  for(int i=0; i<Y_pred.size(); i++){
    loss += -(Y_true[i]*log(Y_pred[i]+eps) + (1-Y_true[i])*log(1-Y_pred[i]+eps));
  }
  loss /= Y_pred.size();
  return loss;
}

std::vector<double> bce_derivative(const std::vector<double>& Y_true, const std::vector<double>& Y_pred){
  std::vector<double> d_bce(Y_pred.size(), 0.0);
  for(int i=0; i<Y_pred.size(); i++){
    d_bce[i] = Y_pred[i] - Y_true[i];
  }
  return d_bce;
}
