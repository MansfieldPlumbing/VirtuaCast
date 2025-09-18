#include "Files.h"
#include "Types.h" 

// FIX: Remove redundant NOMINMAX definition, as it's set globally by CMake.
#include <wincodec.h> // Windows Imaging Component
#include <comdef.h>
#include <fstream>
#include <algorithm>
#include <vector>

#pragma comment(lib, "windowscodecs.lib")
#pragma comment(lib, "ole32.lib")

// Internal helper for COM initialization and cleanup.
struct CoInitGuard {
    bool initialized = false;
    CoInitGuard() {
        HRESULT hr = CoInitializeEx(NULL, COINIT_APARTMENTTHREADED);
        initialized = SUCCEEDED(hr) || hr == RPC_E_CHANGED_MODE;
    }
    ~CoInitGuard() {
        if (initialized) CoUninitialize();
    }
};

namespace VirtuaCast::Files {

    HRESULT ReadImageToBGR(const std::wstring& filepath, WinTegrity::TegrityImageBuffer& out_buffer) {
        if (filepath.empty()) return E_INVALIDARG;
        
        CoInitGuard comGuard;
        if (!comGuard.initialized) return E_FAIL;

        IWICImagingFactory* pFactory = NULL;
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));
        if (FAILED(hr)) return hr;

        IWICBitmapDecoder* pDecoder = NULL;
        hr = pFactory->CreateDecoderFromFilename(filepath.c_str(), NULL, GENERIC_READ, WICDecodeMetadataCacheOnLoad, &pDecoder);
        if (FAILED(hr)) { pFactory->Release(); return hr; }

        IWICBitmapFrameDecode* pFrame = NULL;
        hr = pDecoder->GetFrame(0, &pFrame);
        if (FAILED(hr)) { pDecoder->Release(); pFactory->Release(); return hr; }

        IWICFormatConverter* pConverter = NULL;
        hr = pFactory->CreateFormatConverter(&pConverter);
        if (FAILED(hr)) { pFrame->Release(); pDecoder->Release(); pFactory->Release(); return hr; }

        hr = pConverter->Initialize(pFrame, GUID_WICPixelFormat24bppBGR, WICBitmapDitherTypeNone, NULL, 0.f, WICBitmapPaletteTypeMedianCut);
        if (FAILED(hr)) { pConverter->Release(); pFrame->Release(); pDecoder->Release(); pFactory->Release(); return hr; }

        UINT width, height;
        pConverter->GetSize(&width, &height);

        UINT stride = width * 3;
        UINT bufferSize = stride * height;

        BYTE* buffer = new (std::nothrow) BYTE[bufferSize];
        if (!buffer) { pConverter->Release(); pFrame->Release(); pDecoder->Release(); pFactory->Release(); return E_OUTOFMEMORY; }

        hr = pConverter->CopyPixels(NULL, stride, bufferSize, buffer);
        if (FAILED(hr)) {
            delete[] buffer;
            pConverter->Release(); pFrame->Release(); pDecoder->Release(); pFactory->Release();
            return hr;
        }

        out_buffer.data = buffer;
        out_buffer.width = width;
        out_buffer.height = height;
        out_buffer.channels = 3;

        pConverter->Release(); pFrame->Release(); pDecoder->Release(); pFactory->Release();
        return S_OK;
    }

    HRESULT WriteImage(const std::wstring& filepath, const WinTegrity::TegrityImageBuffer& image) {
        if (filepath.empty() || !image.data) return E_INVALIDARG;

        CoInitGuard comGuard;
        if (!comGuard.initialized) return E_FAIL;
        
        IWICImagingFactory* pFactory = NULL;
        HRESULT hr = CoCreateInstance(CLSID_WICImagingFactory, NULL, CLSCTX_INPROC_SERVER, IID_PPV_ARGS(&pFactory));
        if (FAILED(hr)) return hr;

        std::wstring ext;
        const wchar_t* dot = wcsrchr(filepath.c_str(), L'.');
        if (dot) { ext = dot; std::transform(ext.begin(), ext.end(), ext.begin(), ::towlower); }

        GUID containerFormat = GUID_ContainerFormatBmp;
        bool isJpeg = false;
        if (ext == L".jpg" || ext == L".jpeg") { containerFormat = GUID_ContainerFormatJpeg; isJpeg = true; }

        IWICStream* pStream = NULL;
        hr = pFactory->CreateStream(&pStream);
        if (SUCCEEDED(hr)) hr = pStream->InitializeFromFilename(filepath.c_str(), GENERIC_WRITE);

        IWICBitmapEncoder* pEncoder = NULL;
        if (SUCCEEDED(hr)) hr = pFactory->CreateEncoder(containerFormat, NULL, &pEncoder);
        if (SUCCEEDED(hr)) hr = pEncoder->Initialize(pStream, WICBitmapEncoderNoCache);

        IWICBitmapFrameEncode* pFrameEncode = NULL;
        IPropertyBag2* pPropertybag = NULL;
        if (SUCCEEDED(hr)) hr = pEncoder->CreateNewFrame(&pFrameEncode, &pPropertybag);
        
        if (SUCCEEDED(hr) && isJpeg) {
            PROPBAG2 option = { 0 };
            option.pstrName = const_cast<wchar_t*>(L"ImageQuality");
            VARIANT varValue; VariantInit(&varValue); varValue.vt = VT_R4; varValue.fltVal = 0.95f;
            hr = pPropertybag->Write(1, &option, &varValue); VariantClear(&varValue);
        }
        if (pPropertybag) pPropertybag->Release();

        if (SUCCEEDED(hr)) hr = pFrameEncode->Initialize(NULL);
        if (SUCCEEDED(hr)) hr = pFrameEncode->SetSize(image.width, image.height);

        WICPixelFormatGUID format = GUID_WICPixelFormatUndefined;
        if (image.channels == 3) format = GUID_WICPixelFormat24bppBGR;
        else if (image.channels == 4) format = GUID_WICPixelFormat32bppBGRA;
        else if (image.channels == 1) format = GUID_WICPixelFormat8bppGray;
        else { hr = E_INVALIDARG; }

        if (SUCCEEDED(hr)) hr = pFrameEncode->SetPixelFormat(&format);
        if (SUCCEEDED(hr)) {
            UINT stride = image.width * image.channels;
            UINT bufferSize = stride * image.height;
            hr = pFrameEncode->WritePixels(image.height, stride, bufferSize, image.data);
        }

        if (SUCCEEDED(hr)) hr = pFrameEncode->Commit();
        if (SUCCEEDED(hr)) hr = pEncoder->Commit();

        if (pFrameEncode) pFrameEncode->Release(); if (pEncoder) pEncoder->Release();
        if (pStream) pStream->Release(); if (pFactory) pFactory->Release();

        return SUCCEEDED(hr) ? S_OK : E_FAIL;
    }

    void FreeImageBuffer(WinTegrity::TegrityImageBuffer& buffer) {
        if (buffer.data) {
            delete[] buffer.data;
            buffer.data = nullptr;
            buffer.width = 0;
            buffer.height = 0;
            buffer.channels = 0;
        }
    }

    bool ReadFileToBytes(const std::wstring& filepath, std::vector<char>& out_data) {
        std::ifstream file(filepath, std::ios::binary | std::ios::ate);
        if (!file.is_open()) return false;
        std::streamsize size = file.tellg();
        file.seekg(0, std::ios::beg);
        out_data.resize(static_cast<size_t>(size));
        if (file.read(out_data.data(), size)) return true;
        return false;
    }

    HRESULT ResizeImage(const WinTegrity::TegrityImageBuffer& src, WinTegrity::TegrityImageBuffer& dst) {
        if (!src.data || !dst.data || src.channels != dst.channels) return E_INVALIDARG;
        try {
            double x_ratio = static_cast<double>(src.width - 1) / dst.width;
            double y_ratio = static_cast<double>(src.height - 1) / dst.height;

            for (int y_dst = 0; y_dst < dst.height; ++y_dst) {
                for (int x_dst = 0; x_dst < dst.width; ++x_dst) {
                    double x_src = x_dst * x_ratio, y_src = y_dst * y_ratio;
                    int x1 = static_cast<int>(x_src), y1 = static_cast<int>(y_src);
                    int x2 = std::min(x1 + 1, src.width - 1), y2 = std::min(y1 + 1, src.height - 1);
                    double x_diff = x_src - x1, y_diff = y_src - y1;
                    unsigned char* p_dst = dst.data + (static_cast<size_t>(y_dst) * dst.width + x_dst) * dst.channels;
                    for (int c = 0; c < src.channels; ++c) {
                        const unsigned char* p1 = src.data + (static_cast<size_t>(y1) * src.width + x1) * src.channels + c;
                        const unsigned char* p2 = src.data + (static_cast<size_t>(y1) * src.width + x2) * src.channels + c;
                        const unsigned char* p3 = src.data + (static_cast<size_t>(y2) * src.width + x1) * src.channels + c;
                        const unsigned char* p4 = src.data + (static_cast<size_t>(y2) * src.width + x2) * src.channels + c;
                        double val = (*p1*(1-x_diff)*(1-y_diff)) + (*p2*x_diff*(1-y_diff)) + (*p3*(1-x_diff)*y_diff) + (*p4*x_diff*y_diff);
                        p_dst[c] = static_cast<unsigned char>(val);
                    }
                }
            }
        } catch(...) { return E_FAIL; }
        return S_OK;
    }

    HRESULT DrawRectangle(WinTegrity::TegrityImageBuffer& image, int x1, int y1, int x2, int y2, unsigned char r, unsigned char g, unsigned char b, int thickness) {
        if (!image.data) return E_INVALIDARG;
        for (int i = 0; i < thickness; ++i) {
            for (int x = x1; x <= x2; ++x) { // Top and bottom lines
                if (x >= 0 && x < image.width) {
                    if (y1 + i >= 0 && y1 + i < image.height) { unsigned char* p = image.data + (static_cast<size_t>(y1 + i) * image.width + x) * image.channels; p[0] = b; p[1] = g; p[2] = r; }
                    if (y2 - i >= 0 && y2 - i < image.height) { unsigned char* p = image.data + (static_cast<size_t>(y2 - i) * image.width + x) * image.channels; p[0] = b; p[1] = g; p[2] = r; }
                }
            }
            for (int y = y1; y <= y2; ++y) { // Left and right lines
                if (y >= 0 && y < image.height) {
                    if (x1 + i >= 0 && x1 + i < image.width) { unsigned char* p = image.data + (static_cast<size_t>(y) * image.width + (x1 + i)) * image.channels; p[0] = b; p[1] = g; p[2] = r; }
                    if (x2 - i >= 0 && x2 - i < image.width) { unsigned char* p = image.data + (static_cast<size_t>(y) * image.width + (x2 - i)) * image.channels; p[0] = b; p[1] = g; p[2] = r; }
                }
            }
        }
        return S_OK;
    }

    HRESULT DrawFilledRectangle(WinTegrity::TegrityImageBuffer& image, int x1, int y1, int x2, int y2, unsigned char r, unsigned char g, unsigned char b) {
        if (!image.data) return E_INVALIDARG;
        int start_x = std::max(0, std::min(x1, x2)), end_x = std::min(image.width, std::max(x1, x2));
        int start_y = std::max(0, std::min(y1, y2)), end_y = std::min(image.height, std::max(y1, y2));
        for (int y = start_y; y < end_y; ++y) {
            for (int x = start_x; x < end_x; ++x) {
                unsigned char* p = image.data + (static_cast<size_t>(y) * image.width + x) * image.channels;
                p[0] = b; p[1] = g; p[2] = r;
            }
        }
        return S_OK;
    }
}