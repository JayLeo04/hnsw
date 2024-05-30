#pragma once
#include <iostream>
#include "base.hpp"
#include <vector>
#include <cstring>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <random>
#include <cassert>
#include <cmath>
#include <limits>

namespace HNSWLab {

    class HNSW : public AlgorithmInterface {
    public:
        HNSW(int M, int efConstruction, int m_L);

        void insert(const int *item, int label) override;
        std::vector<int> query(const int *query, int k) override;
        ~HNSW() {}

    private:
        int M;
        int efConstruction;
        int m_L;
        int maxLevel;
        std::mt19937 rng;
        std::uniform_real_distribution<> uniform_dist;

        struct Node {
            int label;
            std::vector<int> data;
            std::vector<std::vector<int>> neighbors; // Neighbors at each level
        };

        std::vector<Node> nodes;
        std::unordered_map<int, int> label_to_index; // Map from label to node index
        std::vector<int> entry_points; // Entry points for each level

        int calculate_level();
        void search_layer(const int *query, std::priority_queue<std::pair<float, int>> &candidates, int level);
        void connect_node(int node_index, int level, std::priority_queue<std::pair<float, int>> &candidates);
        float distance(const int *a, const int *b, size_t size);
    };

    HNSW::HNSW(int M, int efConstruction, int m_L)
        : M(M), efConstruction(efConstruction), m_L(m_L), maxLevel(0),
          rng(std::random_device{}()), uniform_dist(0.0, 1.0) {
        entry_points.push_back(-1); // Initialize entry point at level 0
    }

    int HNSW::calculate_level() {
        return static_cast<int>(-log(uniform_dist(rng)) * m_L);
    }

    float HNSW::distance(const int *a, const int *b, size_t size) {
        float dist = 0.0;
        for (size_t i = 0; i < size; ++i) {
            dist += (a[i] - b[i]) * (a[i] - b[i]);
        }
        return sqrt(dist);
    }

    void HNSW::search_layer(const int *query, std::priority_queue<std::pair<float, int>> &candidates, int level) {
        std::unordered_set<int> visited;
        std::priority_queue<std::pair<float, int>> top_candidates;

        if (entry_points[level] != -1) {
            candidates.push({distance(query, nodes[entry_points[level]].data.data(), nodes[entry_points[level]].data.size()), entry_points[level]});
            top_candidates.push({distance(query, nodes[entry_points[level]].data.data(), nodes[entry_points[level]].data.size()), entry_points[level]});
            visited.insert(entry_points[level]);
        }

        while (!candidates.empty()) {
            auto closest = candidates.top();
            candidates.pop();

            int current_node = closest.second;
            float current_dist = closest.first;

            for (int neighbor : nodes[current_node].neighbors[level]) {
                if (visited.find(neighbor) == visited.end()) {
                    visited.insert(neighbor);

                    float dist = distance(query, nodes[neighbor].data.data(), nodes[neighbor].data.size());
                    if (dist < top_candidates.top().first || top_candidates.size() < efConstruction) {
                        candidates.push({dist, neighbor});
                        top_candidates.push({dist, neighbor});
                        if (top_candidates.size() > efConstruction) {
                            top_candidates.pop();
                        }
                    }
                }
            }
        }

        candidates = top_candidates;
    }

    void HNSW::connect_node(int node_index, int level, std::priority_queue<std::pair<float, int>> &candidates) {
        std::vector<int> neighbors;
        while (!candidates.empty() && neighbors.size() < M) {
            neighbors.push_back(candidates.top().second);
            candidates.pop();
        }

        nodes[node_index].neighbors[level] = neighbors;

        for (int neighbor : neighbors) {
            if (nodes[neighbor].neighbors[level].size() < M) {
                nodes[neighbor].neighbors[level].push_back(node_index);
            } else {
                // If the neighbor list is full, we need to maintain only the closest M neighbors
                std::priority_queue<std::pair<float, int>> neighbor_candidates;
                for (int n : nodes[neighbor].neighbors[level]) {
                    neighbor_candidates.push({distance(nodes[neighbor].data.data(), nodes[n].data.data(), nodes[n].data.size()), n});
                }
                neighbor_candidates.push({distance(nodes[neighbor].data.data(), nodes[node_index].data.data(), nodes[node_index].data.size()), node_index});
                std::vector<int> new_neighbors;
                while (!neighbor_candidates.empty() && new_neighbors.size() < M) {
                    new_neighbors.push_back(neighbor_candidates.top().second);
                    neighbor_candidates.pop();
                }
                nodes[neighbor].neighbors[level] = new_neighbors;
            }
        }
    }

    void HNSW::insert(const int *item, int label) {
        int level = calculate_level();
        if (level > maxLevel) {
            maxLevel = level;
            entry_points.push_back(nodes.size());
        }

        Node newNode{label, std::vector<int>(item, item + 128), std::vector<std::vector<int>>(level + 1)};
        nodes.push_back(newNode);
        label_to_index[label] = nodes.size() - 1;

        std::priority_queue<std::pair<float, int>> candidates;
        for (int l = maxLevel; l > level; --l) {
            search_layer(item, candidates, l);
            if (!candidates.empty()) {
                entry_points[l] = candidates.top().second;
            }
        }

        for (int l = std::min(maxLevel, level); l >= 0; --l) {
            search_layer(item, candidates, l);
            connect_node(nodes.size() - 1, l, candidates);
        }
    }

    std::vector<int> HNSW::query(const int *query, int k) {
        std::vector<int> res;
        std::priority_queue<std::pair<float, int>> candidates;

        search_layer(query, candidates, maxLevel);

        while (!candidates.empty() && res.size() < k) {
            res.push_back(candidates.top().second);
            candidates.pop();
        }

        return res;
    }

}