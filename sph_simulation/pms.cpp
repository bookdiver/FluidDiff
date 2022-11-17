#include <iostream>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <cmath>
#include <omp.h>
#include <time.h>

#include "func.h"
#include "func.cpp"

using namespace std;
// ********** Environment ************
class Environment {
    public:
        int num_particles_in_each_dim;
        int num_particles;

        float viscosity;
        float noise_sigma;

        float boundary_height;
        float boundary_width;

        float kernel_radius;
        float dt;
        float max_t;
        float eps;

        vector<Particle> particles;

        Environment(int num_particles_in_each_dim, 
                    float viscosity, 
                    float noise_sigma, 
                    float boundary_height, 
                    float boundary_width, 
                    float kernel_radius, 
                    float dt, 
                    float max_t,
                    float eps);

        void init_particles(void);
        void buildNeighborList(void);
        void computeDensityAndPressure(void);
        void update(void);
        void clear(void);
        void output(void);
        void simulate(void);
        void multi_simulate(int num_simulations);
};

Environment::Environment(int num_particles_in_each_dim, 
                         float viscosity, 
                         float noise_sigma, 
                         float boundary_height, 
                         float boundary_width, 
                         float kernel_radius, 
                         float dt, 
                         float max_t,
                         float eps) {
    this->num_particles_in_each_dim = num_particles_in_each_dim;
    this->num_particles = num_particles_in_each_dim * num_particles_in_each_dim;

    this->viscosity = viscosity;
    this->noise_sigma = noise_sigma;

    this->boundary_height = boundary_height;
    this->boundary_width = boundary_width;

    this->kernel_radius = kernel_radius;
    this->dt = dt;
    this->max_t = max_t;
    this->eps = eps;
    
    // this->particles.resize(this->num_particles);

    float space = 0.1;      // mm
    float radius = 0.5;     // mm
    float dx = 2 * radius + space;
    float dy = 2 * radius + space;
    for (int i = 0; i < num_particles_in_each_dim; ++i){
        for (int j = 0; j < num_particles_in_each_dim; ++j)
        {
            Particle p;
            p.pos = Vec2(i * dx - (num_particles_in_each_dim - 1) / 2 * dx, j * dy - (num_particles_in_each_dim - 1) / 2 * dy);
            p.vel = Vec2(3.0, 3.0);
            p.mass = 1.0;
            p.rho = 0.0;
            p.pressure = 0.0;
            p.press_force = Vec2(0.0, 0.0);
            p.visc_force = Vec2(0.0, 0.0);
            p.ex_force = Vec2(0.0, 0.0);
            p.acc = Vec2(0.0, 0.0);
            this->particles.push_back(p);
        }
    }
}

void Environment::init_particles(void) {
    float space = 0.1;      // mm
    float radius = 0.5;     // mm
    float dx = 2 * radius + space;
    float dy = 2 * radius + space;
    for (int i = 0; i < num_particles; ++i) {
        particles[i].pos.x = i / num_particles_in_each_dim * dx - (num_particles_in_each_dim - 1) / 2 * dx;
        particles[i].pos.y = i % num_particles_in_each_dim * dy - (num_particles_in_each_dim - 1) / 2 * dy;
        particles[i].vel = Vec2(3.0, 3.0);
        particles[i].acc = Vec2(0.0, 0.0);
        particles[i].press_force = Vec2(0.0, 0.0);
        particles[i].visc_force = Vec2(0.0, 0.0);
        particles[i].ex_force = Vec2(0.0, 0.0);
    }
}

void Environment::clear(void) {
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); ++i){
        particles[i].neighbors.clear();
        particles[i].rho = 0.0;
        particles[i].pressure = 0.0;
        particles[i].press_force = Vec2(0.0, 0.0);
        particles[i].visc_force = Vec2(0.0, 0.0);
        particles[i].ex_force = Vec2(0.0, 0.0);
        particles[i].acc = Vec2(0.0, 0.0);
    }
}

void Environment::buildNeighborList(void){
    #pragma omp parallel for
    for (int i = 0; i < particles.size(); ++i){
        for (int j = 0; j < particles.size(); ++j){
            if (i == j) continue;
            float r = (particles[i].pos - particles[j].pos).len();
            if (r < kernel_radius){
                Neighbor n;
                n.i = j;
                n.r = r;
                particles[i].neighbors.push_back(n);
            }
        }
    }
}

void Environment::computeDensityAndPressure(void){
    #pragma omp parallel for 
    for (int i = 0; i < particles.size(); ++i){
        particles[i].pressure = 340.0 * 1.0 / 7 * (pow(particles[i].rho / 1.0, 7) - 1.0);
        for (int ni = 0; ni < particles[i].neighbors.size(); ++ni){
            int j = particles[i].neighbors[ni].i;
            float r = particles[i].neighbors[ni].r;
            particles[i].rho += particles[j].mass * poly6(r, kernel_radius);
            if (particles[i].rho < eps) particles[i].rho = eps;
        }
    }
}

void Environment::update(void) 
{
    clear();
    buildNeighborList();
    computeDensityAndPressure();

    #pragma omp parallel for
    for (int i = 0; i < particles.size(); ++i)
    {
        for (int ni = 0; ni < particles[i].neighbors.size(); ++ni)
        {
            int j = particles[i].neighbors[ni].i;
            float r = particles[i].neighbors[ni].r;
            particles[i].visc_force += viscosity * particles[j].mass * (particles[j].vel - particles[i].vel) * viscosity_lap(r, kernel_radius) / particles[j].rho;
            particles[i].press_force += -particles[j].mass * (particles[i].pressure + particles[j].pressure) / (2 * particles[j].rho) * spiky_grad(r, kernel_radius) * (particles[j].pos - particles[i].pos) / r;
        }
        particles[i].ex_force = Vec2(randn(0.0, noise_sigma), randn(0.0, noise_sigma));
        particles[i].acc = (particles[i].press_force + particles[i].visc_force + particles[i].ex_force) / particles[i].mass;
        particles[i].vel += particles[i].acc * dt;
        particles[i].pos += particles[i].vel * dt;
        float vel_mag = particles[i].vel.len2();
        // If the velocity is greater than the max velocity, then cut it in half.
        if(vel_mag > 4.f) particles[i].vel = particles[i].vel * .5f;
        if (particles[i].pos.x < - boundary_width / 2)
        {
            particles[i].pos += Vec2(-boundary_width / 2 - particles[i].pos.x, 0.0);
            particles[i].vel -= Vec2(2 * particles[i].vel.x, 0.0);
        }
        if (particles[i].pos.x > boundary_width / 2)
        {
            particles[i].pos += Vec2(boundary_width /2 - particles[i].pos.x, 0.0);
            particles[i].vel -= Vec2(2 * particles[i].vel.x, 0.0);
        }
        if (particles[i].pos.y < -boundary_height / 2)
        {
            particles[i].pos += Vec2(0.0, -boundary_height / 2 - particles[i].pos.y);
            particles[i].vel -= Vec2(0.0, 2 * particles[i].vel.y);
        }
        if (particles[i].pos.y > boundary_height / 2)
        {
            particles[i].pos += Vec2(0.0, boundary_height / 2 - particles[i].pos.y);
            particles[i].vel -= Vec2(0.0, 2 * particles[i].vel.y);
        }
    }
}

void Environment::simulate(void) {
    for (float t = 0; t < max_t; t += dt){
        update();
    }
}

void Environment::multi_simulate(int num_simulations){
    clock_t start, end;
    ofstream myfile;
    myfile.open("./positions.txt");
    // #pragma omp parallel for
    for (int n_sim=0; n_sim < num_simulations; n_sim++){
        start = clock();
        init_particles();
        simulate();
        for (int i = 0; i < particles.size(); ++i)
        {
            myfile << particles[i].pos.x << " " << particles[i].pos.y << endl;
        }
        end = clock();
        cout<<"Simulation "<<n_sim<<" done, "<<"Time used: "<<(double)(end - start)/CLOCKS_PER_SEC<<endl;
    }
    myfile.close();
}

int main(int argc, char** argv)
{
    const int num_particles_in_each_dim = 28;
    const int num_particles = num_particles_in_each_dim * num_particles_in_each_dim;

    const float viscosity = 0.1;
    const float noise_sigma = 1;

    const float boundary_height = 100.0;       // height of the boundary TODO: at this moment, the boundary can be ignored
    const float boundary_width = 100.0;        // width of the boundary

    const float kernel_radius = 2.0;            // radius of the kernel function
    const float dt = 1e-2;
    const float max_t = 2.0;
    const float eps = 1e-3;

    Environment env(num_particles_in_each_dim, viscosity, noise_sigma, boundary_height, boundary_width, kernel_radius, dt, max_t, eps);
    
    env.multi_simulate(1000);
    
    return 0;
}