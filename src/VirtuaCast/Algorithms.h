#pragma once

#define WIN32_LEAN_AND_MEAN

#include <windows.h>
#include <Eigen/Dense>
#include <Eigen/SVD>
#include <vector>
#include <cmath>
#include <algorithm>
#include "Types.h"
#include <omp.h>

namespace WinTegrity {

#pragma region Core Helpers & Algorithms

#ifdef TEGRITY_ENABLE_EROSION
inline void erodeMaskIterations(unsigned char* data, int width, int height, int iterations) {
    if (iterations <= 0) return;
    std::vector<unsigned char> temp(static_cast<size_t>(width) * height);
    unsigned char* src = data;
    unsigned char* dst = temp.data();

    for (int i = 0; i < iterations; ++i) {
        #pragma omp parallel for
        for (int y = 1; y < height - 1; ++y) {
            for (int x = 1; x < width - 1; ++x) {
                size_t idx = static_cast<size_t>(y) * width + x;
                unsigned char min_val = 255;
                min_val = std::min(min_val, src[idx]);
                min_val = std::min(min_val, src[idx - 1]);
                min_val = std::min(min_val, src[idx + 1]);
                min_val = std::min(min_val, src[idx - width]);
                min_val = std::min(min_val, src[idx + width]);
                dst[idx] = min_val;
            }
        }
        std::swap(src, dst);
    }
    if (src != data) {
        memcpy(data, src, static_cast<size_t>(width) * height);
    }
}
#endif
struct BlendingParameters {
    int blur_kernel_radius;
    float blur_sigma;
    int erosion_iterations;
};
inline BlendingParameters tuneBlendingParameters(int face_bbox_width, const TegrityPipelineConfig& config) {
    BlendingParameters params;
    const int min_face_width = 50, max_face_width = 500;
    const float blur_scale = 0.05f, erosion_scale = 0.02f;
    int clamped_width = std::max(min_face_width, std::min(max_face_width, face_bbox_width));
    params.blur_kernel_radius = static_cast<int>(std::round(config.mask_blur_intensity + (clamped_width - min_face_width) * blur_scale));
    params.blur_kernel_radius = std::max(1, params.blur_kernel_radius | 1);
    params.blur_sigma = static_cast<float>(params.blur_kernel_radius) / 2.0f;
    params.erosion_iterations = static_cast<int>(std::round(config.mask_erosion_level + (clamped_width - min_face_width) * erosion_scale));
    params.erosion_iterations = std::max(0, params.erosion_iterations);
    return params;
}
inline void applyGaussianMaskBlur(TegrityImageBuffer& mask, int radius, float sigma) {
    int kernel_size = 2 * radius + 1;
    std::vector<float> kernel(kernel_size);
    float sum = 0.0f;
    for (int i = 0; i < kernel_size; ++i) {
        float x = static_cast<float>(i - radius);
        kernel[i] = std::exp(-(x * x) / (2.0f * sigma * sigma));
        sum += kernel[i];
    }
    for (float& val : kernel) { val /= sum; }
    std::vector<unsigned char> temp_buffer(static_cast<size_t>(mask.width) * mask.height);
    #pragma omp parallel for
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            float weighted_sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sample_x = std::clamp(x + k, 0, mask.width - 1);
                weighted_sum += mask.data[static_cast<size_t>(y) * mask.width + sample_x] * kernel[k + radius];
            }
            temp_buffer[static_cast<size_t>(y) * mask.width + x] = static_cast<unsigned char>(weighted_sum);
        }
    }
    #pragma omp parallel for
    for (int y = 0; y < mask.height; ++y) {
        for (int x = 0; x < mask.width; ++x) {
            float weighted_sum = 0.0f;
            for (int k = -radius; k <= radius; ++k) {
                int sample_y = std::clamp(y + k, 0, mask.height - 1);
                weighted_sum += temp_buffer[static_cast<size_t>(sample_y) * mask.width + x] * kernel[k + radius];
            }
            mask.data[static_cast<size_t>(y) * mask.width + x] = static_cast<unsigned char>(weighted_sum);
        }
    }
}
namespace ConvexHull {
    struct Point { Eigen::Vector2d coord; };
    inline int orientation(Point p, Point q, Point r) {
        double val = (q.coord.y() - p.coord.y()) * (r.coord.x() - q.coord.x()) - (q.coord.x() - p.coord.x()) * (r.coord.y() - q.coord.y());
        if (std::abs(val) < 1e-9) return 0;
        return (val > 0) ? 1 : 2;
    }
    inline std::vector<Point> compute(std::vector<Point>& points) {
        if (points.size() < 3) return {};
        int ymin = 0;
        for (size_t i = 1; i < points.size(); i++) {
            if (points[i].coord.y() < points[ymin].coord.y() || (points[i].coord.y() == points[ymin].coord.y() && points[i].coord.x() < points[ymin].coord.x())) ymin = (int)i;
        }
        std::swap(points[0], points[ymin]);
        std::sort(points.begin() + 1, points.end(), [&](Point p1, Point p2) {
            int o = orientation(points[0], p1, p2);
            if (o == 0) return (points[0].coord - p1.coord).squaredNorm() < (points[0].coord - p2.coord).squaredNorm();
            return (o == 2);
        });
        std::vector<Point> hull;
        for (const auto& p : points) {
            while (hull.size() >= 2 && orientation(hull[hull.size() - 2], hull.back(), p) != 2) hull.pop_back();
            hull.push_back(p);
        }
        return hull;
    }
}
inline void generateConvexHullMask(const std::vector<ConvexHull::Point>& points, TegrityImageBuffer& mask) {
    memset(mask.data, 0, static_cast<size_t>(mask.width) * mask.height);
    if (points.size() < 3) return;
    auto hull = ConvexHull::compute(const_cast<std::vector<ConvexHull::Point>&>(points));
    if (hull.size() < 3) return;
    
    #pragma omp parallel for
    for (int y = 0; y < mask.height; ++y) {
        std::vector<int> intersections;
        for (size_t i = 0; i < hull.size(); ++i) {
            auto p1 = hull[i];
            auto p2 = hull[(i + 1) % hull.size()];
            if ((p1.coord.y() <= y && p2.coord.y() > y) || (p2.coord.y() <= y && p1.coord.y() > y)) {
                double ix = p1.coord.x() + (static_cast<double>(y) - p1.coord.y()) * (p2.coord.x() - p1.coord.x()) / (p2.coord.y() - p1.coord.y());
                intersections.push_back(static_cast<int>(std::round(ix)));
            }
        }
        std::sort(intersections.begin(), intersections.end());
        for (size_t i = 0; i + 1 < intersections.size(); i += 2) {
            for (int x = intersections[i]; x < intersections[i + 1]; ++x) {
                if (x >= 0 && x < mask.width) mask.data[static_cast<size_t>(y) * mask.width + x] = 255;
            }
        }
    }
}
inline void applyColorTransfer(const TegrityImageBuffer& target, const TegrityRect& target_roi, TegrityImageBuffer& source_patch) {
    long target_pixel_count = 0;
    Eigen::Vector3d target_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d target_std_dev_sq = Eigen::Vector3d::Zero();
    for (int y = target_roi.y; y < target_roi.y + target_roi.height; ++y) {
        for (int x = target_roi.x; x < target_roi.x + target_roi.width; ++x) {
            if (x < 0 || x >= target.width || y < 0 || y >= target.height) continue;
            const unsigned char* p = target.data + (static_cast<size_t>(y) * target.width + x) * target.channels;
            target_mean.x() += p[0]; target_mean.y() += p[1]; target_mean.z() += p[2];
            target_pixel_count++;
        }
    }
    if (target_pixel_count < 25) { 
        return;
    }
    target_mean /= target_pixel_count;
    for (int y = target_roi.y; y < target_roi.y + target_roi.height; ++y) {
        for (int x = target_roi.x; x < target_roi.x + target_roi.width; ++x) {
            if (x < 0 || x >= target.width || y < 0 || y >= target.height) continue;
            const unsigned char* p = target.data + (static_cast<size_t>(y) * target.width + x) * target.channels;
            target_std_dev_sq.x() += (p[0] - target_mean.x()) * (p[0] - target_mean.x());
            target_std_dev_sq.y() += (p[1] - target_mean.y()) * (p[1] - target_mean.y());
            target_std_dev_sq.z() += (p[2] - target_mean.z()) * (p[2] - target_mean.z());
        }
    }
    Eigen::Vector3d target_std_dev = (target_std_dev_sq / target_pixel_count).array().sqrt();
    long source_pixel_count = static_cast<long>(source_patch.width) * source_patch.height;
    if (source_pixel_count == 0) return;
    Eigen::Map<Eigen::Matrix<unsigned char, Eigen::Dynamic, 3, Eigen::RowMajor>> source_map(source_patch.data, source_pixel_count, 3);
    Eigen::Matrix<double, Eigen::Dynamic, 3> source_matrix_d = source_map.cast<double>();
    Eigen::Vector3d source_mean = source_matrix_d.colwise().mean();
    Eigen::Vector3d source_std_dev = ((source_matrix_d.rowwise() - source_mean.transpose()).array().square().colwise().sum() / source_pixel_count).sqrt();
    for (int i = 0; i < 3; ++i) { if (source_std_dev(i) < 1e-5) source_std_dev(i) = 1.0; }
    source_matrix_d.rowwise() -= source_mean.transpose();
    source_matrix_d.array().rowwise() /= source_std_dev.transpose().array();
    source_matrix_d.array().rowwise() *= target_std_dev.transpose().array();
    source_matrix_d.rowwise() += target_mean.transpose();
    source_map = source_matrix_d.cwiseMax(0.0).cwiseMin(255.0).cast<unsigned char>();
}
inline Eigen::Matrix<double, 2, 3> invertAffineTransform(const Eigen::Matrix<double, 2, 3>& M) {
    Eigen::Matrix2d R = M.leftCols<2>(); Eigen::Vector2d t = M.rightCols<1>();
    Eigen::Matrix2d R_inv = R.inverse(); Eigen::Vector2d t_inv = -R_inv * t;
    Eigen::Matrix<double, 2, 3> M_inv; M_inv.leftCols<2>() = R_inv; M_inv.rightCols<1>() = t_inv;
    return M_inv;
}
inline Eigen::Matrix<double, 2, 3> estimateSimilarityTransform(const Eigen::Matrix<double, 5, 2>& src_pts, const Eigen::Matrix<double, 5, 2>& dst_pts) {
    Eigen::MatrixXd src_mat = src_pts.transpose();
    Eigen::MatrixXd dst_mat = dst_pts.transpose();
    Eigen::Vector2d src_mean = src_mat.rowwise().mean();
    Eigen::Vector2d dst_mean = dst_mat.rowwise().mean();
    Eigen::MatrixXd src_demean = src_mat.colwise() - src_mean;
    Eigen::MatrixXd dst_demean = dst_mat.colwise() - dst_mean;
    Eigen::MatrixXd H = src_demean * dst_demean.transpose();
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(H, Eigen::ComputeThinU | Eigen::ComputeThinV);
    Eigen::MatrixXd U = svd.matrixU();
    Eigen::MatrixXd V = svd.matrixV();
    Eigen::MatrixXd R = V * U.transpose();
    if (R.determinant() < 0) {
        V.col(1) *= -1;
        R = V * U.transpose();
    }
    double var_src = src_demean.array().pow(2).sum() / src_demean.cols();
    Eigen::MatrixXd D = Eigen::MatrixXd::Identity(H.rows(), H.cols());
    if (H.determinant() < 0) {
        D(D.rows()-1, D.cols()-1) = -1;
    }
    double scale = (svd.singularValues().array() * D.diagonal().array()).sum() / (var_src * src_demean.cols());
    Eigen::Vector2d t = dst_mean - scale * R * src_mean;
    Eigen::Matrix<double, 2, 3> M;
    M.leftCols<2>() = scale * R;
    M.rightCols<1>() = t;
    return M;
}
inline void warpAffineBilinear(const TegrityImageBuffer& src, TegrityImageBuffer& dst, const Eigen::Matrix<double, 2, 3>& M, const TegrityRect* roi) {
    Eigen::Matrix<double, 2, 3> M_inv = invertAffineTransform(M);
    int start_y = roi ? std::max(0, roi->y) : 0;
    int end_y = roi ? std::min(dst.height, roi->y + roi->height) : dst.height;
    int start_x = roi ? std::max(0, roi->x) : 0;
    int end_x = roi ? std::min(dst.width, roi->x + roi->width) : dst.width;
    #pragma omp parallel for
    for (int y_dst = start_y; y_dst < end_y; ++y_dst) {
        for (int x_dst = start_x; x_dst < end_x; ++x_dst) {
            double x_src = M_inv(0, 0) * x_dst + M_inv(0, 1) * y_dst + M_inv(0, 2);
            double y_src = M_inv(1, 0) * x_dst + M_inv(1, 1) * y_dst + M_inv(1, 2);
            unsigned char* dst_pixel = dst.data + (static_cast<size_t>(y_dst) * dst.width + x_dst) * dst.channels;
            if (x_src < 0 || x_src >= src.width -1 || y_src < 0 || y_src >= src.height -1) { 
                for(int c=0; c < dst.channels; ++c) dst_pixel[c] = 0;
                continue; 
            }
            int x1 = static_cast<int>(x_src), y1 = static_cast<int>(y_src);
            int x2 = x1 + 1; int y2 = y1 + 1;
            double x_frac = x_src - x1, y_frac = y_src - y1;
            const unsigned char* p1 = src.data + (static_cast<size_t>(y1) * src.width + x1) * src.channels; 
            const unsigned char* p2 = src.data + (static_cast<size_t>(y1) * src.width + x2) * src.channels;
            const unsigned char* p3 = src.data + (static_cast<size_t>(y2) * src.width + x1) * src.channels; 
            const unsigned char* p4 = src.data + (static_cast<size_t>(y2) * src.width + x2) * src.channels;
            double inv_x_frac = 1.0 - x_frac, inv_y_frac = 1.0 - y_frac;
            for (int c = 0; c < dst.channels; ++c) {
                double top = p1[c] * inv_x_frac + p2[c] * x_frac;
                double bottom = p3[c] * inv_x_frac + p4[c] * x_frac;
                dst_pixel[c] = static_cast<unsigned char>(std::clamp(top * inv_y_frac + bottom * y_frac, 0.0, 255.0));
            }
        }
    }
}
inline void blendAlpha(TegrityImageBuffer& background, const TegrityImageBuffer& foreground, const TegrityImageBuffer& alpha_mask, const TegrityRect* roi) {
    int start_y = roi ? std::max(0, roi->y) : 0;
    int end_y = roi ? std::min(background.height, roi->y + roi->height) : background.height;
    int start_x = roi ? std::max(0, roi->x) : 0;
    int end_x = roi ? std::min(background.width, roi->x + roi->width) : background.width;
    int roi_h = end_y - start_y;
    int roi_w = end_x - start_x;
    if (roi_h <= 0 || roi_w <= 0) return;

    using Stride = Eigen::Stride<Eigen::Dynamic, Eigen::Dynamic>;
    Eigen::Map<Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor>, Eigen::Aligned, Stride>
        bg_map(background.data + (static_cast<size_t>(start_y) * background.width + start_x) * background.channels, roi_h, roi_w * background.channels, Stride(background.width * background.channels, background.channels));

    Eigen::Map<const Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor>, Eigen::Aligned, Stride>
        fg_map(foreground.data + (static_cast<size_t>(start_y) * foreground.width + start_x) * foreground.channels, roi_h, roi_w * foreground.channels, Stride(foreground.width * foreground.channels, foreground.channels));

    Eigen::Map<const Eigen::Matrix<unsigned char, -1, -1, Eigen::RowMajor>, Eigen::Aligned, Stride>
        alpha_map(alpha_mask.data + (static_cast<size_t>(start_y) * alpha_mask.width + start_x), roi_h, roi_w, Stride(alpha_mask.width, 1));

    Eigen::ArrayXXf bg_f = bg_map.cast<float>();
    Eigen::ArrayXXf fg_f = fg_map.cast<float>();
    Eigen::ArrayXXf alpha_f = alpha_map.cast<float>() / 255.0f;

    for (int c = 0; c < background.channels; ++c) {
        bg_f.col(c) = fg_f.col(c) * alpha_f.col(0) + bg_f.col(c) * (1.0f - alpha_f.col(0));
    }

    bg_map = bg_f.cwiseMax(0.f).cwiseMin(255.f).cast<unsigned char>();
}
inline std::wstring ToWide(const std::string& narrow) {
    if (narrow.empty()) return std::wstring();
    int wideLen = MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, nullptr, 0);
    if (wideLen == 0) return L"";
    std::wstring wide(wideLen, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, narrow.c_str(), -1, &wide[0], wideLen);
    wide.pop_back();
    return wide;
}
inline std::string ToNarrow(const std::wstring& wide) {
    if (wide.empty()) return std::string();
    int narrowLen = WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, nullptr, 0, NULL, NULL);
    if (narrowLen == 0) return "";
    std::string narrow(narrowLen, '\0');
    WideCharToMultiByte(CP_UTF8, 0, wide.c_str(), -1, &narrow[0], narrowLen, NULL, NULL);
    narrow.pop_back();
    return narrow;
}
inline TegrityRect get_face_roi(const TegrityDetectedFace& face, int img_width, int img_height) {
    float x1 = face.bbox[0]; float y1 = face.bbox[1]; float x2 = face.bbox[2]; float y2 = face.bbox[3];
    float box_w = x2 - x1; float box_h = y2 - y1; float center_x = x1 + box_w / 2.0f; float center_y = y1 + box_h / 2.0f;
    float long_side = std::max(box_w, box_h);
    int final_side = static_cast<int>(long_side * 1.5f);
    TegrityRect roi;
    roi.x = static_cast<int>(center_x - final_side / 2.0f);
    roi.y = static_cast<int>(center_y - final_side / 2.0f);
    roi.width = final_side;
    roi.height = final_side;
    return roi;
}
inline void normalize_l2(float* data, int size) {
    Eigen::Map<Eigen::VectorXf> v(data, size);
    float norm = v.norm();
    if (norm > 1e-5f) v /= norm;
}
#pragma endregion

} // namespace WinTegrity