// File: src/VirtuaCast/SourceFace.cpp

#include "SourceFace.h"
#include "FaceDetection.h"
#include "FaceEmbedding.h"
#include "FaceSwap.h"
#include "OnnxRuntime.h"
#include "Files.h"
#include "Console.h"
#include "Types.h"
#include <filesystem>
#include <fstream>
#include <d3d11.h>
#include <wrl/client.h>

namespace VirtuaCast {

struct SourceFaceManager::Impl {
    std::string m_sources_directory;
    OnnxRuntime* m_onnx = nullptr;
    FaceDetector* m_detector = nullptr;
    FaceEmbedder* m_embedder = nullptr;
    Console* m_console = nullptr;

    std::vector<SourceFace> m_source_faces;
    int m_current_index = -1;

    HRESULT GenerateAndSaveEmbedding(const std::wstring& image_path, const std::wstring& emb_path);
    bool LoadEmbedding(const std::wstring& emb_path, std::vector<float>& out_embedding);
    bool SaveEmbedding(const std::wstring& emb_path, const std::vector<float>& embedding);
};

SourceFaceManager::SourceFaceManager() : pImpl(std::make_unique<Impl>()) {}
SourceFaceManager::~SourceFaceManager() = default;

HRESULT SourceFaceManager::Initialize(
    const std::string& sources_directory,
    OnnxRuntime& onnx,
    FaceDetector& detector,
    FaceEmbedder& embedder,
    Console* console
) {
    pImpl->m_sources_directory = sources_directory;
    pImpl->m_onnx = &onnx;
    pImpl->m_detector = &detector;
    pImpl->m_embedder = &embedder;
    pImpl->m_console = console;

    if (!std::filesystem::exists(pImpl->m_sources_directory)) {
        if (pImpl->m_console) pImpl->m_console->AddLog("[ERROR] Sources directory not found: %s", pImpl->m_sources_directory.c_str());
        return E_INVALIDARG;
    }

    try {
        for (const auto& entry : std::filesystem::directory_iterator(pImpl->m_sources_directory)) {
            if (entry.is_regular_file()) {
                std::filesystem::path path = entry.path();
                std::wstring ext = path.extension().wstring();
                std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower);

                if (ext == L".jpg" || ext == L".jpeg" || ext == L".png" || ext == L".webp") {
                    std::filesystem::path emb_path = path;
                    emb_path.replace_extension(L".emb");

                    SourceFace face;
                    face.image_path = path.wstring();
                    face.display_name = path.stem().string();

                    if (std::filesystem::exists(emb_path)) {
                        if (pImpl->LoadEmbedding(emb_path.wstring(), face.embedding)) {
                             pImpl->m_source_faces.push_back(std::move(face));
                             if (pImpl->m_console) pImpl->m_console->AddLog("Loaded source: %s", face.display_name.c_str());
                        } else {
                            if (pImpl->m_console) pImpl->m_console->AddLog("[WARNING] Corrupt embedding file, regenerating: %s", emb_path.filename().string().c_str());
                            if (SUCCEEDED(pImpl->GenerateAndSaveEmbedding(path.wstring(), emb_path.wstring()))) {
                                pImpl->LoadEmbedding(emb_path.wstring(), face.embedding);
                                pImpl->m_source_faces.push_back(std::move(face));
                            }
                        }
                    } else {
                        if (pImpl->m_console) pImpl->m_console->AddLog("Generating embedding for: %s", face.display_name.c_str());
                        if (SUCCEEDED(pImpl->GenerateAndSaveEmbedding(path.wstring(), emb_path.wstring()))) {
                             pImpl->LoadEmbedding(emb_path.wstring(), face.embedding);
                             pImpl->m_source_faces.push_back(std::move(face));
                        }
                    }
                }
            }
        }
    } catch (const std::filesystem::filesystem_error& e) {
        if (pImpl->m_console) pImpl->m_console->AddLog("[ERROR] Filesystem error: %s", e.what());
        return E_FAIL;
    }
    
    if (!pImpl->m_source_faces.empty()) {
        pImpl->m_current_index = 0;
    }

    return S_OK;
}


HRESULT SourceFaceManager::Impl::GenerateAndSaveEmbedding(const std::wstring& image_path, const std::wstring& emb_path) {
    WinTegrity::TegrityImageBuffer image_bgr;
    HRESULT hr = Files::ReadImageToBGR(image_path, image_bgr);
    if (FAILED(hr)) {
        if (m_console) m_console->AddLog("[ERROR] Failed to read image file: %S", image_path.c_str());
        return hr;
    }
    
    Microsoft::WRL::ComPtr<ID3D11Device> device;
    Microsoft::WRL::ComPtr<ID3D11DeviceContext> context;
    D3D_FEATURE_LEVEL featureLevel;
    hr = D3D11CreateDevice(nullptr, D3D_DRIVER_TYPE_HARDWARE, nullptr, 0, nullptr, 0, D3D11_SDK_VERSION, &device, &featureLevel, &context);
    if(FAILED(hr)) {
        Files::FreeImageBuffer(image_bgr);
        return hr;
    }

    D3D11_TEXTURE2D_DESC desc = {};
    desc.Width = image_bgr.width;
    desc.Height = image_bgr.height;
    desc.MipLevels = 1;
    desc.ArraySize = 1;
    desc.Format = DXGI_FORMAT_B8G8R8A8_UNORM;
    desc.SampleDesc.Count = 1;
    desc.Usage = D3D11_USAGE_DEFAULT;
    desc.BindFlags = D3D11_BIND_SHADER_RESOURCE;

    std::vector<uint8_t> image_bgra( (size_t)image_bgr.width * image_bgr.height * 4);
    for(int i = 0; i < image_bgr.width * image_bgr.height; ++i) {
        image_bgra[(size_t)i*4+0] = image_bgr.data[(size_t)i*3+0];
        image_bgra[(size_t)i*4+1] = image_bgr.data[(size_t)i*3+1];
        image_bgra[(size_t)i*4+2] = image_bgr.data[(size_t)i*3+2];
        image_bgra[(size_t)i*4+3] = 255;
    }

    D3D11_SUBRESOURCE_DATA subData = {};
    subData.pSysMem = image_bgra.data();
    subData.SysMemPitch = image_bgr.width * 4;

    Microsoft::WRL::ComPtr<ID3D11Texture2D> texture;
    hr = device->CreateTexture2D(&desc, &subData, &texture);
    if(FAILED(hr)) { Files::FreeImageBuffer(image_bgr); return hr; }
    
    Microsoft::WRL::ComPtr<ID3D11ShaderResourceView> srv;
    hr = device->CreateShaderResourceView(texture.Get(), nullptr, &srv);
    if(FAILED(hr)) { Files::FreeImageBuffer(image_bgr); return hr; }
    
    FrameData frameData;
    frameData.pDevice = device.Get();
    frameData.pContext = context.Get();
    frameData.pInputSRV = srv.Get();
    frameData.pConsole = m_console;
    
    m_detector->Process(frameData);
    if (frameData.faces.empty()) {
        if (m_console) m_console->AddLog("[WARNING] No face found in source image: %S", image_path.c_str());
        Files::FreeImageBuffer(image_bgr);
        return E_FAIL;
    }
    
    m_embedder->Process(frameData);
    Files::FreeImageBuffer(image_bgr);
    
    if (frameData.faces.front().embedding.empty()) {
        if (m_console) m_console->AddLog("[ERROR] Failed to generate embedding for: %S", image_path.c_str());
        return E_FAIL;
    }
    
    if (SaveEmbedding(emb_path, frameData.faces.front().embedding)) {
        return S_OK;
    }
    
    return E_FAIL;
}

bool SourceFaceManager::Impl::LoadEmbedding(const std::wstring& emb_path, std::vector<float>& out_embedding) {
    std::ifstream file(emb_path, std::ios::binary);
    if (!file.is_open()) return false;
    file.seekg(0, std::ios::end);
    size_t size = file.tellg();
    file.seekg(0, std::ios::beg);
    if (size != 512 * sizeof(float)) return false;
    out_embedding.resize(512);
    file.read(reinterpret_cast<char*>(out_embedding.data()), size);
    return true;
}

bool SourceFaceManager::Impl::SaveEmbedding(const std::wstring& emb_path, const std::vector<float>& embedding) {
    if (embedding.size() != 512) return false;
    std::ofstream file(emb_path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return false;
    file.write(reinterpret_cast<const char*>(embedding.data()), 512 * sizeof(float));
    return true;
}


void SourceFaceManager::CycleSource(int delta) {
    if (pImpl->m_source_faces.empty()) return;

    pImpl->m_current_index += delta;

    if (pImpl->m_current_index >= (int)pImpl->m_source_faces.size()) {
        pImpl->m_current_index = 0;
    }
    if (pImpl->m_current_index < 0) {
        pImpl->m_current_index = (int)pImpl->m_source_faces.size() - 1;
    }
}

const SourceFace* SourceFaceManager::GetCurrentSource() const {
    if (pImpl->m_current_index < 0 || pImpl->m_current_index >= (int)pImpl->m_source_faces.size()) {
        return nullptr;
    }
    return &pImpl->m_source_faces[pImpl->m_current_index];
}


void SourceFaceManager::SetCurrentSourceAsActive(FaceSwap& swapper) {
    const SourceFace* current = GetCurrentSource();
    if (current) {
        swapper.SetSourceFace(current->embedding);
        if (pImpl->m_console) pImpl->m_console->AddLog("Set active source face to: %s", current->display_name.c_str());
    } else {
        swapper.ClearSourceFace();
        if (pImpl->m_console) pImpl->m_console->AddLog("Cleared active source face.");
    }
}

}