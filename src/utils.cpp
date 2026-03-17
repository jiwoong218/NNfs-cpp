#include <vector>
#include <cmath>
#include <random>

std::vector<double> sigmoid(const std::vector<double>& u){
  std::vector<double> result(u.size(), 0.0);
  for(int i=0; i<u.size(); i++){
     result[i] = 1/(1+exp(-u[i]));
  }
  return result; // h
}
std::vector<double> sigmoid_derivative(const std::vector<double>& h){
  std::vector<double> result(h.size(), 0.0);
  for(int i=0; i<h.size(); i++){
    result[i] = h[i] * (1-h[i]);
  }
  return result; // d_sigmoid
}

double xavier_uniform(int fan_in, int fan_out){
  static std::random_device rd;
  static std::mt19937 gen(rd());
  double limit = std::sqrt(6.0 / (fan_in + fan_out));
  std::uniform_real_distribution<double> dist(-limit, limit);
  return dist(gen);
}
