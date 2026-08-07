#include <iostream>
#include <sstream>
#include <string>
#include <stdexcept>
#include <vector>
#include "computation/sequence_converter.h"

void printUsage() {
    std::cout << "Usage:\n"
              << "  sequence_to_bt --yaml <path/to/sequence.yaml> --out <path/to/output.xml> [--section <name>] [--arm <name>]\n"
              << "  sequence_to_bt --yaml <path/to/sequence.yaml> --out <path/to/output.xml> --both-arms \\\n"
              << "      --left-section <name> --right-section <name> [--profile <ompl_profile>] [--tree-name <name>] \\\n"
              << "      [--home-section <name>] [--repeat-cycles N] [--exclude-points N,N,...]\n";
}

int main(int argc, char* argv[]) {
    std::string yaml_path = "";
    std::string out_path = "";
    std::string section = "";
    std::string arm = "";
    bool both_arms = false;
    std::string left_section = "";
    std::string right_section = "";
    std::string profile = "realtime_rrt";
    std::string tree_name = "Both";
    std::string home_section = "";
    int repeat_cycles = 1;
    std::vector<int> exclude_points;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--yaml") {
            if (i + 1 < argc) {
                yaml_path = argv[++i];
            } else {
                std::cerr << "[ERROR] --yaml requires a value\n";
                return 1;
            }
        } else if (arg == "--out") {
            if (i + 1 < argc) {
                out_path = argv[++i];
            } else {
                std::cerr << "[ERROR] --out requires a value\n";
                return 1;
            }
        } else if (arg == "--section") {
            if (i + 1 < argc) {
                section = argv[++i];
            } else {
                std::cerr << "[ERROR] --section requires a value\n";
                return 1;
            }
        } else if (arg == "--arm") {
            if (i + 1 < argc) {
                arm = argv[++i];
            } else {
                std::cerr << "[ERROR] --arm requires a value\n";
                return 1;
            }
        } else if (arg == "--both-arms") {
            both_arms = true;
        } else if (arg == "--left-section") {
            if (i + 1 < argc) {
                left_section = argv[++i];
            } else {
                std::cerr << "[ERROR] --left-section requires a value\n";
                return 1;
            }
        } else if (arg == "--right-section") {
            if (i + 1 < argc) {
                right_section = argv[++i];
            } else {
                std::cerr << "[ERROR] --right-section requires a value\n";
                return 1;
            }
        } else if (arg == "--profile") {
            if (i + 1 < argc) {
                profile = argv[++i];
            } else {
                std::cerr << "[ERROR] --profile requires a value\n";
                return 1;
            }
        } else if (arg == "--tree-name") {
            if (i + 1 < argc) {
                tree_name = argv[++i];
            } else {
                std::cerr << "[ERROR] --tree-name requires a value\n";
                return 1;
            }
        } else if (arg == "--home-section") {
            if (i + 1 < argc) {
                home_section = argv[++i];
            } else {
                std::cerr << "[ERROR] --home-section requires a value\n";
                return 1;
            }
        } else if (arg == "--repeat-cycles") {
            if (i + 1 < argc) {
                repeat_cycles = std::stoi(argv[++i]);
            } else {
                std::cerr << "[ERROR] --repeat-cycles requires a value\n";
                return 1;
            }
        } else if (arg == "--exclude-points") {
            if (i + 1 < argc) {
                std::stringstream ss(argv[++i]);
                std::string token;
                while (std::getline(ss, token, ',')) {
                    if (!token.empty()) {
                        exclude_points.push_back(std::stoi(token));
                    }
                }
            } else {
                std::cerr << "[ERROR] --exclude-points requires a value\n";
                return 1;
            }
        } else if (arg == "--help" || arg == "-h") {
            printUsage();
            return 0;
        } else {
            std::cerr << "[ERROR] Unknown argument: " << arg << "\n";
            printUsage();
            return 1;
        }
    }

    if (yaml_path.empty()) {
        std::cerr << "[ERROR] Missing required argument: --yaml\n";
        printUsage();
        return 1;
    }
    if (out_path.empty()) {
        std::cerr << "[ERROR] Missing required argument: --out\n";
        printUsage();
        return 1;
    }
    if (both_arms && (left_section.empty() || right_section.empty())) {
        std::cerr << "[ERROR] --both-arms requires --left-section and --right-section\n";
        printUsage();
        return 1;
    }

    try {
        if (both_arms) {
            utilities::computation::convertBothArmsSequenceToBt(
                yaml_path, out_path, left_section, right_section, profile, tree_name, home_section, repeat_cycles,
                exclude_points);
        } else {
            utilities::computation::convertSequenceToBt(yaml_path, out_path, section, arm);
        }
        std::cout << "[OK] Written " << out_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
