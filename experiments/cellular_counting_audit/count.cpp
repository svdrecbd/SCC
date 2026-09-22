// Exact component counting control with explicit time and recursion limits.
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cstdint>
#include <iomanip>
struct Integer {
    std::vector<uint32_t> words;
    Integer(uint32_t value = 0) { if (value) words.push_back(value); }
    void normalize() { while (!words.empty() && words.back() == 0) words.pop_back(); }
    Integer operator+(const Integer &other) const {
        Integer result; uint64_t carry = 0;
        for (size_t index = 0; index < std::max(words.size(), other.words.size()) || carry; ++index) {
            uint64_t sum = carry + (index < words.size() ? words[index] : 0ULL)
                          + (index < other.words.size() ? other.words[index] : 0ULL);
            result.words.push_back(static_cast<uint32_t>(sum)); carry = sum >> 32;
        }
        result.normalize(); return result;
    }
    Integer operator*(const Integer &other) const {
        Integer result; result.words.resize(words.size() + other.words.size(), 0);
        for (size_t left = 0; left < words.size(); ++left) {
            uint64_t carry = 0;
            for (size_t right = 0; right < other.words.size(); ++right) {
                uint64_t product = uint64_t(words[left]) * other.words[right] + result.words[left+right] + carry;
                result.words[left+right] = static_cast<uint32_t>(product); carry = product >> 32;
            }
            if (!other.words.empty()) result.words[left+other.words.size()] = static_cast<uint32_t>(carry);
        }
        result.normalize(); return result;
    }
    Integer &operator*=(const Integer &other) { *this = *this * other; return *this; }
    Integer operator<<(long bits) const {
        Integer result; result.words.resize(bits / 32, 0); uint64_t carry = 0;
        for (uint32_t word : words) { uint64_t shifted = (uint64_t(word) << (bits % 32)) | carry;
            result.words.push_back(static_cast<uint32_t>(shifted)); carry = shifted >> 32; }
        if (carry) result.words.push_back(static_cast<uint32_t>(carry));
        result.normalize(); return result;
    }
    bool operator==(uint32_t value) const { return value == 0 ? words.empty() : words.size() == 1 && words[0] == value; }
    friend std::ostream &operator<<(std::ostream &output, Integer value) {
        if (value.words.empty()) return output << '0';
        std::vector<uint32_t> decimal;
        while (!value.words.empty()) {
            uint64_t remainder = 0;
            for (size_t index = value.words.size(); index-- > 0;) {
                uint64_t dividend = (remainder << 32) | value.words[index];
                value.words[index] = static_cast<uint32_t>(dividend / 1000000000);
                remainder = dividend % 1000000000;
            }
            decimal.push_back(static_cast<uint32_t>(remainder)); value.normalize();
        }
        output << decimal.back();
        for (size_t index = decimal.size() - 1; index-- > 0;) output << std::setfill('0') << std::setw(9) << decimal[index];
        return output;
    }
};
using Clause = std::vector<int>;
using Formula = std::vector<Clause>;
struct ResourceLimit {};

struct Counter {
    std::chrono::steady_clock::time_point deadline;
    std::string strategy;
    int variable_count;
    long calls = 0;
    long maximum_calls;
    std::map<Formula, Integer> cache;
    void check() {
        if (calls > maximum_calls || std::chrono::steady_clock::now() > deadline)
            throw ResourceLimit();
    }
    static void canonical(Formula &clauses) {
        Formula result;
        for (auto clause : clauses) {
            std::sort(clause.begin(), clause.end());
            clause.erase(std::unique(clause.begin(), clause.end()), clause.end());
            bool tautology = false;
            for (int literal : clause)
                if (std::binary_search(clause.begin(), clause.end(), -literal)) tautology = true;
            if (!tautology) result.push_back(clause);
        }
        std::sort(result.begin(), result.end());
        result.erase(std::unique(result.begin(), result.end()), result.end());
        clauses.swap(result);
    }
    Clause support(const Formula &clauses) {
        std::vector<bool> present(variable_count + 1, false);
        for (const auto &clause : clauses) for (int literal : clause) present[std::abs(literal)] = true;
        Clause result;
        for (int variable = 1; variable <= variable_count; ++variable) if (present[variable]) result.push_back(variable);
        return result;
    }
    Integer count(Formula clauses, const Clause &variables) {
        ++calls; check();
        std::vector<int> assigned(variable_count + 1, 0);
        int assigned_count = 0;
        while (true) {
            check();
            bool added = false;
            for (const auto &clause : clauses) {
                if (clause.empty()) return 0;
                if (clause.size() == 1) {
                    int literal = clause.front(), variable = std::abs(literal), value = literal > 0 ? 1 : -1;
                    if (assigned[variable] == -value) return 0;
                    if (!assigned[variable]) { assigned[variable] = value; ++assigned_count; added = true; }
                }
            }
            if (!added) break;
            Formula reduced;
            for (const auto &clause : clauses) {
                Clause remainder;
                bool satisfied = false;
                for (int literal : clause) {
                    int value = assigned[std::abs(literal)];
                    if (value == (literal > 0 ? 1 : -1)) { satisfied = true; break; }
                    if (value == 0) remainder.push_back(literal);
                }
                if (!satisfied) reduced.push_back(remainder);
            }
            clauses.swap(reduced);
        }
        Clause active = support(clauses);
        long free_count = static_cast<long>(variables.size()) - assigned_count - active.size();
        if (free_count < 0) throw std::runtime_error("Invalid free variable count");
        Integer factor = Integer(1) << free_count;
        if (clauses.empty()) return factor;
        canonical(clauses); check();
        auto cached = cache.find(clauses);
        if (cached != cache.end()) return factor * cached->second;
        Clause parents(variable_count + 1); std::iota(parents.begin(), parents.end(), 0);
        auto root = [&](int variable) { while (parents[variable] != variable) {
            parents[variable] = parents[parents[variable]]; variable = parents[variable]; } return variable; };
        for (const auto &clause : clauses) for (int literal : clause)
            parents[root(std::abs(literal))] = root(std::abs(clause.front()));
        std::map<int, Formula> groups;
        for (const auto &clause : clauses) groups[root(std::abs(clause.front()))].push_back(clause);
        Integer result = 0;
        if (groups.size() > 1) {
            result = 1;
            for (const auto &entry : groups) { result *= count(entry.second, support(entry.second)); if (result == 0) break; }
        } else {
            int variable = active.back();
            if (strategy == "occurrence") {
                Clause frequencies(variable_count + 1, 0);
                for (const auto &clause : clauses) for (int literal : clause) ++frequencies[std::abs(literal)];
                for (int candidate : active) if (frequencies[candidate] >= frequencies[variable]) variable = candidate;
            }
            Formula positive = clauses; positive.push_back({variable});
            Formula negative = clauses; negative.push_back({-variable});
            result = count(positive, active) + count(negative, active);
        }
        cache.emplace(clauses, result);
        return factor * result;
    }
};

int main(int argc, char **argv) {
    if (argc != 5) return 2;
    auto started = std::chrono::steady_clock::now();
    std::ifstream input(argv[1]); if (!input) return 2;
    Formula clauses; Clause pending; std::string line; int variables = -1, expected = -1;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == 'c') continue;
        std::istringstream stream(line);
        if (line[0] == 'p') { std::string marker, kind; stream >> marker >> kind >> variables >> expected;
            if (kind != "cnf") return 2; }
        else { int literal; while (stream >> literal) {
            if (literal == 0) { clauses.push_back(pending); pending.clear(); } else pending.push_back(literal); } }
    }
    if (variables < 0 || !pending.empty() || static_cast<int>(clauses.size()) != expected) return 2;
    for (const auto &clause : clauses) for (int literal : clause) if (std::abs(literal) > variables) return 2;
    Counter counter; counter.variable_count = variables; counter.strategy = argv[2];
    counter.maximum_calls = std::stol(argv[4]);
    counter.deadline = started + std::chrono::milliseconds(static_cast<long>(1000 * std::stod(argv[3])));
    std::string status = "complete"; Integer result = 0;
    try { Counter::canonical(clauses); Clause scope(variables); std::iota(scope.begin(), scope.end(), 1);
        result = counter.count(clauses, scope); }
    catch (ResourceLimit &) { status = "resource_limit"; }
    double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    std::cout << "{\"status\":\"" << status << "\",\"count\":";
    if (status == "complete") std::cout << result; else std::cout << "null";
    std::cout << ",\"seconds\":" << elapsed << ",\"recursive_calls\":" << counter.calls
              << ",\"cached_components\":" << counter.cache.size() << "}\n";
}
