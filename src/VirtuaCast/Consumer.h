// File: src/VirtuaCast/Consumer.h

#pragma once

#include <windows.h>
#include <d3d11_4.h>
#include <wrl/client.h>
#include <string>
#include <vector>
#include <memory>

struct ID3D11Device;
struct ID3D11DeviceContext;
struct ID3D11ShaderResourceView;

namespace VirtuaCast {

    struct DiscoveredSharedStream {
        DWORD processId;
        std::wstring processName;
        std::wstring producerType;
        std::wstring manifestName;
        std::wstring textureName;
        std::wstring fenceName;
        LUID adapterLuid;
    };

    class Consumer {
    public:
        Consumer();
        ~Consumer();

        HRESULT Initialize(ID3D11Device* device, ID3D11DeviceContext* context);
        void Teardown();

        void DiscoverStreams();
        HRESULT Connect(const DiscoveredSharedStream& stream_info);
        void Disconnect();
        ID3D11ShaderResourceView* UpdateAndGetSRV();

        bool IsConnected() const;
        const std::vector<DiscoveredSharedStream>& GetDiscoveredStreams() const;
        const DiscoveredSharedStream* GetActiveStreamInfo() const;
        std::wstring GetProducerName() const;
        UINT GetWidth() const;
        UINT GetHeight() const;

    private:
        struct Impl;
        std::unique_ptr<Impl> pImpl;
    };

}