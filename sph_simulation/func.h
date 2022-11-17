#ifndef FUNC_H
#define FUNC_H

float poly6(float r, float kernel_radius);
float spiky_grad(float r, float kernel_radius);
float viscosity_lap(float r, float kernel_radius);
float randn(float mu, float sigma);

struct Vec2;
struct Neighbor;
struct Particle;

#endif