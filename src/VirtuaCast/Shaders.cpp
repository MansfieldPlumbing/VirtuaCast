// File: src/VirtuaCast/Shaders.cpp

#include "Shaders.h"
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <map>

namespace VirtuaCast {

// PIMPL struct holds the shader cache.
struct ShaderManager::Impl {
    std::map<std::string, std::vector<char>> m_shaderCache;
};

// --- Method Implementations ---

ShaderManager::ShaderManager() : pImpl(std::make_unique<Impl>()) {}
ShaderManager::~ShaderManager() { Teardown(); }

HRESULT ShaderManager::Initialize(const std::string& shader_directory) {
    pImpl->m_shaderCache.clear();

    if (!std::filesystem::exists(shader_directory)) {
        return E_INVALIDARG; // Or a more specific error
    }

    try {
        for (const auto& entry : std::filesystem::directory_iterator(shader_directory)) {
            if (entry.is_regular_file() && entry.path().extension() == ".cso") {
                std::string filename = entry.path().filename().string();

                std::ifstream file(entry.path(), std::ios::binary | std::ios::ate);
                if (!file.is_open()) {
                    continue; // Skip if we can't open
                }

                std::streamsize size = file.tellg();
                file.seekg(0, std::ios::beg);

                std::vector<char> buffer(size);
                if (file.read(buffer.data(), size)) {
                    pImpl->m_shaderCache[filename] = std::move(buffer);
                }
            }
        }
    }
    catch (const std::filesystem::filesystem_error& e) {
        // Handle filesystem errors (e.g., permissions)
        return E_FAIL;
    }

    return S_OK;
}

void ShaderManager::Teardown() {
    pImpl->m_shaderCache.clear();
}

const std::vector<char>& ShaderManager::GetShaderBytecode(const std::string& shader_name) const {
    auto it = pImpl->m_shaderCache.find(shader_name);
    if (it == pImpl->m_shaderCache.end()) {
        throw std::runtime_error("Shader not found in cache: " + shader_name);
    }
    return it->second;
}

} // namespace VirtuaCast