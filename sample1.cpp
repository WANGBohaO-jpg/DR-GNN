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
    std::vector<int> result;
    std::vector<int> pool;
    for (int i = 0; i <= N; ++i) {
        if (pos_set.find(i) == pos_set.end()) {
            pool.push_back(i);
        }
    }
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> distr(0, pool.size() - 1);
    for (int i = 0; i < num; ++i) {
        result.push_back(pool[distr(gen)]);
    }
    return result;
}


PYBIND11_MODULE(sample1, m) {
    m.def("choose_items", &choose_items);
}
/*
<%
setup_pybind11(cfg)
%>
*/