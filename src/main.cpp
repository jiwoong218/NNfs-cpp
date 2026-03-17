#include <iostream>
#include <vector>

#include "network.cpp"

int main() {
  std::vector<std::vector<double>> X_train = {
    {0.0, 0.0}, 
    {0.0, 1.0}, 
    {1.0, 0.0}, 
    {1.0, 1.0}
  };
  std::vector<std::vector<double>> Y_train = {
    {0.0}, 
    {1.0}, 
    {1.0}, 
    {0.0}
  };

  Network net;
  net.add_layer(2, 4);
  net.add_layer(4, 1);

  double learning_rate = 0.3;

  for(int epoch=0; epoch<3000; epoch++){
    double total_loss = 0.0;
    for (int i=0; i<X_train.size(); i++){
      std::vector<double> Y_pred = net.forward(X_train[i]);
      total_loss += bce(Y_train[i], Y_pred);
      net.backward(Y_train[i], Y_pred, learning_rate);
    }

    if ((epoch + 1) % 100 == 0) {
      std::cout << "Epoch: " << epoch + 1 
        << "| Loss: " << total_loss / X_train.size()
        << std::endl;
    }
  }

  std::cout << "\n---Trained Network---" << std::endl;
  for (int i = 0; i < X_train.size(); i++) {
    std::vector<double> Y_pred = net.forward(X_train[i]);
    std::cout << "Input: " << X_train[i][0] << ", " << X_train[i][1]
              << " | Pred: " << Y_pred[0] << " | True: " << Y_train[i][0]
              << std::endl;
  }
  return 0;
}
