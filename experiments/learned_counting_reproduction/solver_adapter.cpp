// Preserve the original search while exposing its existing branching callback.
#include "solver.h"
#include <cstring>
#include <vector>
#include <exception>
#include <chrono>
using PolicyCallback = int (*)(unsigned, const unsigned *, const unsigned *, unsigned, const double *);
extern "C" int count_models(const char *path, PolicyCallback callback, int seconds,
                            char *count_output, unsigned output_capacity,
                            unsigned long *decisions, unsigned long *calls) {
    SharpSAT::Solver solver;
    solver.config().quiet = true;
    solver.statistics().maximum_cache_size_bytes_ = 512ULL * 1024 * 1024;
    solver.setTimeBound(seconds);
    int selected = -1;
    *calls = 0;
    auto *oracle = new SharpSAT::BranchingOracle(
        [](CVIG) {},
        [&](CVIG graph) -> int * {
            ++*calls;
            selected = -1;
            if (callback) {
                auto labels = solver.getLitLabels();
                std::vector<double> flattened;
                flattened.reserve(labels.size()*4);
                for (const auto &row : labels) flattened.insert(flattened.end(),row.begin(),row.end());
                const auto &rows = std::get<0>(graph);
                const auto &columns = std::get<1>(graph);
                selected = callback(rows.size(), rows.data(), columns.data(), labels.size(), flattened.data());
                if (selected < -1) { solver.interrupt(); selected = -1; }
            }
            return &selected;
        });
    solver.setBranchingOracle(oracle);
    try { solver.solve(path); }
    catch (...) { solver.setBranchingOracle(nullptr); return -2; }
    *decisions = solver.statistics().num_decisions_;
    int status = solver.statistics().exit_state_ == SharpSAT::SUCCESS ? 0 : 1;
    if (status == 0) {
        std::string result = solver.statistics().final_solution_count().get_str();
        if (result.size()+1 > output_capacity) status = -3;
        else std::memcpy(count_output,result.c_str(),result.size()+1);
    }
    solver.setBranchingOracle(nullptr);
    return status;
}
