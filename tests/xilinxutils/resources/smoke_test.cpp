// smoke_test.cpp
#include <iostream>
#include <vector>

int main() {
    // Print a unique string that Python can search for in the logs
    std::cout << "--- PYSILICON SMOKE TEST START ---" << std::endl;
    
    // Test a basic C++ container to ensure the HLS toolchain is fully indexed
    std::vector<int> test_vec = {1, 2, 3, 4, 5};
    int sum = 0;
    for(int i : test_vec) {
        sum += i;
    }

    if (sum == 15) {
        std::cout << "Logic Check: PASSED" << std::endl;
        std::cout << "--- PYSILICON SMOKE TEST END ---" << std::endl;
        return 0; // Success
    } else {
        std::cerr << "Logic Check: FAILED (Sum mismatch)" << std::endl;
        return 1; // Failure
    }
}