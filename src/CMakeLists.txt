# B:/VirtuaCast/src/DirectPort/CMakeLists.txt

cmake_minimum_required(VERSION 3.15)
project(DirectPort LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

#======================================================================
# 1. PYTHON MODULE ("directport.pyd")
#======================================================================
find_package(pybind11 CONFIG REQUIRED)
add_library(DirectPortLib STATIC "DirectPort.cpp")
target_include_directories(DirectPortLib PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(DirectPortLib PUBLIC d3d11.lib d3d12.lib dxgi.lib advapi32.lib user32.lib Synchronization.lib d3dcompiler.lib)
pybind11_add_module(directport "pybind_wrapper.cpp")
target_link_libraries(directport PRIVATE DirectPortLib)
message(STATUS "Configured DirectPort C++ library and Python module.")

#======================================================================
# 2. STANDALONE EXECUTABLE ("DirectPortDemultiplexer.exe")
#======================================================================
add_executable(DirectPortDemultiplexer WIN32
    "DirectPortDemultiplexer.cpp"
    "resource.rc"
)
target_link_libraries(DirectPortDemultiplexer PRIVATE d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib advapi32.lib gdi32.lib)
message(STATUS "Configured DirectPortDemultiplexer standalone executable.")

#======================================================================
# 3. STANDALONE EXECUTABLE ("DirectPortProducerD3D11.exe")
#======================================================================
add_executable(DirectPortProducerD3D11 WIN32
    "DirectPortProducerD3D11.cpp"
    "resource.rc"
)
target_link_libraries(DirectPortProducerD3D11 PRIVATE d3d11.lib d3d12.lib dxgi.lib d3dcompiler.lib user32.lib advapi32.lib gdi32.lib)
message(STATUS "Configured DirectPortProducerD3D11 standalone executable.")

#======================================================================
# 4. STANDALONE EXECUTABLE ("DirectPortProducerD3D12.exe")
#    This is the new D3D12 Producer target
#======================================================================
add_executable(DirectPortProducerD3D12 WIN32
    "DirectPortProducerD3D12.cpp"
    "resource.rc"
)
target_link_libraries(DirectPortProducerD3D12 PRIVATE d3d12.lib dxgi.lib d3dcompiler.lib user32.lib advapi32.lib Synchronization.lib gdi32.lib)
message(STATUS "Configured DirectPortProducerD3D12 standalone executable.")

#======================================================================
# 5. STANDALONE EXECUTABLE ("DirectPortCamera.exe")
#    This is the new Camera target
#======================================================================
add_executable(DirectPortCamera WIN32
    "DirectPortCamera.cpp"
    "resource.rc"
)
# Note the extra libraries for Media Foundation
target_link_libraries(DirectPortCamera PRIVATE
    d3d12.lib dxgi.lib d3dcompiler.lib user32.lib advapi32.lib gdi32.lib
    mf.lib mfplat.lib mfreadwrite.lib mfuuid.lib shlwapi.lib Synchronization.lib dxguid.lib
)
message(STATUS "Configured DirectPortCamera standalone executable.")

#-------------------------------------------------------------------------------
# 6. VirtuaCamCamera (Hosted Camera Producer DLL)
#-------------------------------------------------------------------------------
set(CAMERAMODULE_SOURCE_FILES DirectPortVirtuaCam/CameraModule.cpp)
add_library(VirtuaCamCamera SHARED ${CAMERAMODULE_SOURCE_FILES})
set_target_properties(VirtuaCamCamera PROPERTIES OUTPUT_NAME "DirectPortCamera")
target_compile_definitions(VirtuaCamCamera PRIVATE _WINDOWS _USRDLL UNICODE _UNICODE CAMERAMODULE_EXPORTS)
target_link_libraries(VirtuaCamCamera PRIVATE
    mfplat mfuuid mfreadwrite d3d12 dxgi d3dcompiler advapi32
)