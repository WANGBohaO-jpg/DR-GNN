/*
<%
cfg['compiler_args'] = ['-std=c++11']
cfg['include_dirs'] = ['path/to/your/include/files']
setup_pybind11(cfg)
%>
*/

#include <pybind11/stl.h>
#include <pybind11/pybind11.h>
#include <vector>
#include <set>
#include <random>
#include<iostream>
#include<pybind11/pybind11.h>
#include<pybind11/numpy.h>
#include<algorithm>
#include<vector>
#include<cmath>
using namespace std;
namespace py = pybind11;


std::vector<int> choose_items(int N, int num, const std::unordered_set<int>& pos_set) {
    std::random_device rd;
    std::mt19937 g(rd());

    std::vector<int> result;
    for (int i = 0; i < num; i++) {
        int random_number;
        do {
            std::uniform_int_distribution<> distr(0, N);
            random_number = distr(g);
        } while (pos_set.find(random_number) != pos_set.end());
        result.push_back(random_number);
    }

    return result;
}

PYBIND11_MODULE(sample2, m) {
    m.def("choose_items", &choose_items);
}
/*
<%
setup_pybind11(cfg)
%>
*/