#pragma once

struct NormalizationParams {
    int hidden;
    int rows;
    float epsilon;
};
struct GroupNormParams { int hidden; int rows; float epsilon; int groups; };
