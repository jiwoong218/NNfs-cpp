#include "layer.cpp"

class Network{
public:
  std::vector<Layer> layers;

  void add_layer(int input_size, int output_size){
    layers.push_back(Layer(input_size, output_size));
  }

  std::vector<double> forward(std::vector<double> X){
    for(int i=0; i<layers.size(); i++){
      X = layers[i].forward(X);
    }
    return X; // h, Y_pred
  }

  void backward(const std::vector<double>& Y_true, const std::vector<double>& Y_pred, double learning_rate){
    std::vector<double> current_delta = layers[layers.size()-1].calculate_delta(Y_true, Y_pred); // 마지막 단의 delta, W를 필요로 하지 않음.
    for(int i=layers.size()-1; i>=0; i--){
      std::vector<std::vector<double>> W = layers[i].backward(current_delta, learning_rate); 
      if(i>0){
        std::vector<double> next_delta(layers[i].h_prev.size(), 0.0); // 한 레이어 앞 단의 delta 구하기, delta의 개수는 해당 레이어의 출력 개수와 같음
        std::vector<double> d_sigmoid = sigmoid_derivative(layers[i].h_prev); 
        for(int j=0; j<layers[i].b.size(); j++){
          for(int k=0; k<W[0].size(); k++){
            next_delta[k] += current_delta[j] * W[j][k] * d_sigmoid[k]; // current_delta: output.size(), W: output.size() * input.size(), d_sigmoid: input.size()
          }
        }
        current_delta = next_delta;
      }
    }
  }
};
