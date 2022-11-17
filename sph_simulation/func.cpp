#include<cmath>
#include <cstdlib>
#include <vector>

#include "func.h"

#define pi 3.1415926535897932384626433832795

using namespace std;


// kernel functions
float poly6(float r, float kernel_radius) {
    if (r < kernel_radius && r > 0) {
        float K = 4.0 / (pi * pow(kernel_radius, 8));
        float x = pow(kernel_radius, 2) - pow(r, 2);
        return K * pow(x, 3);
    }
    else return 0.0;
};

// calculate the first derivative of the spiky kernel function
float spiky_grad(float r, float kernel_radius) {
    if (r < kernel_radius && r > 0) {
        float K = - 30.0 / (pi * pow(kernel_radius, 5));
        float x = kernel_radius - r;
        return K * pow(x, 2);
    }
    else return 0.0;
};

// calculate the laplacian of the viscosity kernel function
float viscosity_lap(float r, float kernel_radius) {
    if (r < kernel_radius && r > 0) {
        float K = 20.0 / (pi * pow(kernel_radius, 5));
        float x = kernel_radius - r;
        return - K * x;
    }
    else return 0.0;
};

// generate a gaussian random number
float randn(float mu, float sigma) {
    // Box-Muller transform to generate a gaussian random number
    float x1, x2, w, y1;
    static float y2;
    static int use_last = 0;

    if (use_last) {
        y1 = y2;
        use_last = 0;
    }
    else {
        do {
            x1 = 2.0 * rand() / RAND_MAX - 1.0;
            x2 = 2.0 * rand() / RAND_MAX - 1.0;
            w = x1 * x1 + x2 * x2;
        } while (w >= 1.0);

        w = sqrt((-2.0 * log(w)) / w);
        y1 = x1 * w;
        y2 = x2 * w;
        use_last = 1;
    }

    return (mu + y1 * sigma);
};

// ********** 2D vector **********
struct Vec2 {
    float x, y;
    Vec2() : x(0), y(0) {}
    Vec2(float x, float y) : x(x), y(y) {}
    Vec2 operator+(const Vec2& v) const { return Vec2(x + v.x, y + v.y); }
    Vec2 operator-(const Vec2& v) const { return Vec2(x - v.x, y - v.y); }
    Vec2 operator=(const Vec2& v) { x = v.x; y = v.y; return *this; }
    Vec2 operator+=(const Vec2& v) { return *this = *this + v; }
    Vec2 operator-=(const Vec2& v) { return *this = *this - v; }

    float operator*(const Vec2& v) const { return x * v.x + y * v.y; }
    Vec2 operator*(float s) const { return Vec2(x * s, y * s); }
    Vec2 operator/(float s) const { return Vec2(x / s, y / s); }
    float len2() const { return *this * *this; }
    float len() const { return sqrt(len2()); }
    Vec2 norm() const { return *this / len(); }
};
Vec2 operator*(float s, const Vec2& v) { return Vec2(v.x * s, v.y * s); }

struct Neighbor { 
    int i, j; 
    float r, r2; 
};

struct Particle {
    Vec2 pos, vel, acc, press_force, visc_force, ex_force;
    float mass, rho, pressure;
    vector<Neighbor> neighbors;
};