// File: src/VirtuaCast/CameraSource.h

#pragma once

#include <windows.h>
#include <d3d11_4.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <memory>

struct ID3D11Device;
struct ID3D11ShaderResourceView;

namespace VirtuaCast {

    struct CameraInfo {
        int id;
        std::wstring name;
        std::wstring symbolic_link;
    };

    class CameraSource {
    public:
        CameraSource();
        ~CameraSource();

        HRESULT Initialize(ID3D11Device* pD3D11Device);
        void Teardown();

        const std::vector<CameraInfo>& GetAvailableCameras() const;
        HRESULT ActivateCamera(int cameraId, UINT width = 0, UINT height = 0);
        void DeactivateCamera();
        void SetMirror(bool mirror);

        ID3D11ShaderResourceView* GetFrameSRV();
        
        bool IsActive() const;
        void GetCurrentResolution(UINT& width, UINT& height) const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}