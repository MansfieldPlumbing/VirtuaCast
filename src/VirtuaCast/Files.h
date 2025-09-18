#pragma once

#include <string>
#include <vector>
#include <windows.h> // For HRESULT
#include "Types.h"   // For TegrityImageBuffer

namespace VirtuaCast::Files {
    // Reads an image file into a standard BGR buffer.
    HRESULT ReadImageToBGR(const std::wstring& filepath, WinTegrity::TegrityImageBuffer& out_buffer);
    
    // Writes a BGR, BGRA, or grayscale image buffer to a file (supports .jpg, .bmp).
    HRESULT WriteImage(const std::wstring& filepath, const WinTegrity::TegrityImageBuffer& image);

    // Frees the memory allocated by ReadImageToBGR.
    void FreeImageBuffer(WinTegrity::TegrityImageBuffer& buffer);

    // Reads an entire file into a byte vector (useful for loading models).
    bool ReadFileToBytes(const std::wstring& filepath, std::vector<char>& out_data);

    // Resizes a source image into a destination buffer using bilinear interpolation.
    HRESULT ResizeImage(const WinTegrity::TegrityImageBuffer& src, WinTegrity::TegrityImageBuffer& dst);

    // Draws a rectangle outline on an image buffer.
    HRESULT DrawRectangle(WinTegrity::TegrityImageBuffer& image, int x1, int y1, int x2, int y2, 
                          unsigned char r, unsigned char g, unsigned char b, int thickness);

    // Draws a filled rectangle on an image buffer.
    HRESULT DrawFilledRectangle(WinTegrity::TegrityImageBuffer& image, int x1, int y1, int x2, int y2, 
                                unsigned char r, unsigned char g, unsigned char b);
}