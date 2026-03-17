#include <vector>

#include "utils.cpp"
#include "loss_func.cpp"

class Layer{
public:
  std::vector<double> h_prev; // prev output == input data
  std::vector<std::vector<double>> W;
  std::vector<double> b;
  std::vector<double> u; // before activation

  Layer(int input_size, int output_size){
    // input_size: prev_layer size
    // output_size: next_layer size
    W.resize(output_size, std::vector<double> (input_size));
    for(int i=0; i<output_size; i++){
      for(int j=0; j<input_size; j++){
        W[i][j] = xavier_uniform(input_size, output_size) * 0.1;
      }
    }
    b.resize(output_size, 0.0);
  }

  std::vector<double> forward(const std::vector<double>& X){
    h_prev = X;
    u.resize(b.size(), 0.0);
    for(int i=0; i<b.size(); i++){
      for(int j=0; j<X.size(); j++){
        u[i] += X[j]*W[i][j];
      }
      u[i] += b[i];
    }
    return sigmoid(u); // h
  }

  std::vector<std::vector<double>> backward(const std::vector<double>& delta, double learning_rate){
    // update
    std::vector<std::vector<double>> gradient(W.size(), std::vector<double>(W[0].size(), 0.0));
    for(int i=0; i<b.size(); i++){
      for(int j=0; j<h_prev.size(); j++){
        gradient[i][j] = delta[i] * h_prev[j]; 
        W[i][j] -= learning_rate * gradient[i][j];
      }
      b[i] -= learning_rate * delta[i];
    }
    return W;
  }
  
  std::vector<double> calculate_delta(const std::vector<double>& Y_true, const std::vector<double>& Y_pred){
    std::vector<double> delta(Y_pred.size(), 0.0);
    std::vector<double> d_bce = bce_derivative(Y_true, Y_pred);
    std::vector<double> d_sigmoid = sigmoid_derivative(sigmoid(u));
    for(int i=0; i<Y_pred.size(); i++){
      delta[i] = d_bce[i] * d_sigmoid[i];
    }
    return delta;
  }
};
