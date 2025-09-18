// File: src/VirtuaCast/Shaders.h

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <windows.h> // For HRESULT

namespace VirtuaCast {

    // Manages loading and caching of compiled shader objects (.cso) from disk.
    class ShaderManager {
    public:
        ShaderManager();
        ~ShaderManager();

        // Scans a directory and loads all .cso files found within it into memory.
        HRESULT Initialize(const std::string& shader_directory);
        void Teardown();

        // Retrieves the raw bytecode for a previously loaded shader.
        // Throws a std::runtime_error if the shader is not found.
        const std::vector<char>& GetShaderBytecode(const std::string& shader_name) const;

    private:
        // PIMPL idiom to hide the internal map and filesystem logic.
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

} // namespace VirtuaCast