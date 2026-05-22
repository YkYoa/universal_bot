#include <iostream>
#include <string>
#include <stdexcept>
#include "computation/sequence_converter.h"

void printUsage() {
    std::cout << "Usage:\n"
              << "  sequence_to_bt --yaml <path/to/sequence.yaml> --out <path/to/output.xml> [--section <name>] [--arm <name>]\n";
}

int main(int argc, char* argv[]) {
    std::string yaml_path = "";
    std::string out_path = "";
    std::string section = "";
    std::string arm = "";

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

    try {
        utilities::computation::convertSequenceToBt(yaml_path, out_path, section, arm);
        std::cout << "[OK] Written " << out_path << "\n";
    } catch (const std::exception& e) {
        std::cerr << "[ERROR] " << e.what() << "\n";
        return 1;
    }

    return 0;
}
