#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include "routing_common.h"

int main(int argc, char** argv) {
    if (argc != 3) return 2;
    int k = std::stoi(argv[1]), initial_index = std::stoi(argv[2]);
    if (k < 1 || k > areno_accel::routing::kMaxTopK) return 2;
    std::vector<float> values(k, -INFINITY);
    std::vector<int> indices(k, initial_index);
    std::string item;
    int index = 0;
    while (std::cin >> item) {
        areno_accel::routing::insert_topk(std::stof(item), index++, values.data(), indices.data(), k);
    }
    for (int i : indices) std::cout << i << ' ';
    std::cout << '\n';
}
