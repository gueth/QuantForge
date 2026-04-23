#include <iostream>
#include <cmath>
#include <random>
#include <numeric>
#include <vector>
#include <chrono>

double mc_call_price(double S0, double K, double r, double sigma, double T, int n_paths){

    // Générateur de nombre aléatoires
    std::mt19937_64 rng(42);
    std::normal_distribution<double> norm(0.0, 1.0);

    double sum_payoffs = 0.0;

    for(int i = 0; i < n_paths; i++){
        double Z = norm(rng);
        //1. Calcule ST avec le mouvement brownien
        double ST = S0 * exp((r - 0.5 * pow(sigma, 2)) * T + sigma * Z);
        //2. Calcule le payoff max(ST-K, 0)
        double payoff =std::max(ST -K, 0.0);
        sum_payoffs += payoff;
    }

    //3. retoune la moyenne actualisée
    return exp(-r * T) * (sum_payoffs / n_paths);
}

int main() {
    // warm up
    mc_call_price(100, 100, 0.05, 0.2, 1.0, 100000);

    // Mesure
    auto start = std::chrono::high_resolution_clock::now();

    double price = 0.0;
    for(int i = 0; i < 10; i++){
     price = mc_call_price(100, 100, 0.05, 0.2, 1.0, 100000);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(end -start).count() / 10;

    std::cout << "Prix Monte Carlo : " << price << std::endl;
    std::cout << "Temps moyen : " << ms << " ms par appel" << std::endl;

    return 0;
}