#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <bits/stdc++.h>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include "excel.h"

// forward: input * primera matriz y backward(input)
using namespace boost::numeric::ublas;

enum Function
{
    SIGMOIDE,
    TANH,
    RELU
};

matrix<double> getMatrix(int const &nRows, int const &nCols)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(1, 10);

    matrix<double> init(nRows, nCols);
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            // init(i, j) = distr(eng);
            init(i, j) = 1;
        }
    }
    return init;
}

class MLP
{
private:
    vector<double> input;
    std::vector<int> nodosh;
    int output;
    std::vector<matrix<double>> matrices;
    std::vector<std::vector<double>> soutouts;
    std::vector<double> error;
    std::vector<int> sds;
    Function act_function;

public:
    MLP(std::vector<double> &input, std::vector<int> &nodosh, std::vector<int> &sds, Function func = SIGMOIDE)
    {
        // this->input;
        this->input = vector<double>(input.size());
        std::copy(input.begin(), input.end(), this->input.begin());
        this->nodosh = nodosh;
        this->output = sds.size();
        this->sds = sds;
        this->act_function = func;

        // First layer
        matrix<double> init = getMatrix(input.size(), nodosh[0]);
        matrices.push_back(init);
        // Intermediate layers
        for (int i = 1; i < nodosh.size(); ++i)
        {
            matrix<double> tmp = getMatrix(nodosh[i - 1], nodosh[i]);
            matrices.push_back(tmp);
        }
        // Output Layer
        matrix<double> moutput = getMatrix(nodosh[nodosh.size() - 1], this->output);
        matrices.push_back(moutput);
    }

    vector<double> forward()
    {
        int niveles_h = nodosh.size();
        vector<double> hk_prev = this->input;

        for (int i = 0; i < matrices.size(); ++i)
        {
            hk_prev = prod(hk_prev, matrices[i]);
            std::vector<double> ss(hk_prev.size());
            for (int j = 0; j < hk_prev.size(); j++)
            {
                ss[j] = hk_prev[j] = handle_activation_func(hk_prev[j]);
            }
            soutouts.push_back(ss);
            // std::cout << "hk_final" << i << ": " << hk_prev << std::endl;
        }
        return hk_prev;
    }

    double handle_activation_func(double x)
    {

        if (this->act_function == SIGMOIDE)
        {
            return 1.0 / (1.0 + exp(-x));
        }
        else if (this->act_function == TANH)
        {
            return tanh(x);
        }
        else if (this->act_function == RELU)
        {
            return std::max(0.0, x);
        }
        else
        {
            return x;
        }

        // return switch (this->act_function) {
        //     case SIGMOIDE: return this->sigmoide(x);
        //     case TANH: return this->tanh(x);
        //     case RELU: return this->relu(x);
        // }
    }

    double sigmoide(double x)
    {
        return 1 / (1 + std::exp(-x));
    }

    double tanh(double x)
    {
        return (std::exp(x) - std::exp(-x)) / (std::exp(x) + std::exp(-x));
    }

    double relu(double x)
    {
        return std::max(0.0, x);
    }

    void trainning(double epoch, double alpha)
    {

        for (int i = 0; i < epoch; i++)
        {
            // forwards
            auto eOuput = this->forward();
            // calcular error y guardar el error;
            auto currentError = this->MSE(eOuput);
            std::cout << currentError << std::endl;
            // backwards
            std::cout << "AcualW:";
            printMatrix(this->matrices);
            auto newWeighs = this->backward(alpha);
            // update weights
            //  this->matrices = newWeighs;
            //  this->updateW(newWeighs);
            //  std::cout<<"Update:";
            //  printMatrix(this->matrices);
        }
    }

    std::vector<matrix<double>> backward(double alpha)
    {

        auto matricres_cp = matrices;  // copia de matrices
        std::vector<double> curDeltas; // current Deltas
        for (int i = matrices.size() - 1; i >= 0; i--)
        {
            // std::cout<<i<<std::endl;
            auto currentM = matrices[i];    // matriz de Pesos original
            auto changeM = matricres_cp[i]; // matriz de Pesos temporal
            auto curOuputs = soutouts[i];   // salida actual.
            std::vector<double> outputsK;   // salida anterior.
            if (matrices.size() - 1 == i)
            // hk con ouput
            {
                auto tmpL = soutouts[i - 1];
                outputsK.clear();
                outputsK.resize(tmpL.size());
                std::copy(tmpL.begin(), tmpL.end(), outputsK.begin());
                // precalculo de los deltas
                for (unsigned j = 0; j < output; j++)
                {
                    curDeltas.push_back((curOuputs[j] - sds[j]) * curOuputs[j] * (1 - curOuputs[j]));
                }
                // actualizar los pesos changeM
                for (int j = 0; j < changeM.size1(); j++)
                {
                    for (int k = 0; k < changeM.size2(); k++)
                    {
                        changeM(j, k) = currentM(j, k) - alpha * (curDeltas[k] * outputsK[j]);
                    }
                }
                matricres_cp[i] = changeM;
            }
            else
            {
                if (i == 0)
                {
                    // input con h1
                    outputsK.clear();
                    outputsK.resize(input.size());
                    std::copy(input.begin(), input.end(), outputsK.begin());
                }
                else
                {
                    // hk-1 con hk;
                    outputsK.clear();
                    auto tmpO = soutouts[i + 1];
                    outputsK.resize(tmpO.size());
                    std::copy(tmpO.begin(), tmpO.end(), outputsK.begin());
                }
                // variable delta temporal
                std::vector<double> tmpdelta;
                for (int k = 0; k < currentM.size2(); k++)
                {
                    double deltaTmp;
                    // sumatoria sobre los deltas anteriores.
                    for (int l = 0; l < curDeltas.size(); l++)
                    {
                        deltaTmp += curDeltas[l] * matrices[i + 1](k, l);
                    }
                    // calculo de deltas.
                    deltaTmp *= curOuputs[k] * (1 - curOuputs[k]);
                    tmpdelta.push_back(deltaTmp);
                }

                // actualizar los pesos changeM
                for (int j = 0; j < currentM.size1(); j++)
                {
                    for (int k = 0; k < currentM.size2(); k++)
                    {
                        changeM(j, k) = currentM(j, k) - alpha * (tmpdelta[k] * outputsK[j]);
                    }
                }
                matricres_cp[i] = changeM;
                // cambiar el nuevo Curdelta
                curDeltas.clear();
                curDeltas.resize(tmpdelta.size());
                curDeltas = tmpdelta;
            }
        }
        this->matrices = matricres_cp;
        return matricres_cp;
    }

    void updateW(std::vector<matrix<double>> current)
    {
        this->matrices = current;
    }

    double MSE(vector<double> output)
    {
        double vError = 0;
        for (unsigned i = 0; i < output.size(); i++)
        {
            vError += (output[i] - sds[i]) * (output[i] - sds[i]);
        }
        vError = vError / output.size();
        error.push_back(vError);
        return vError;
    }

    void print(std::vector<double> item)
    {
        for (auto i : item)
            std::cout << i << " ";
        std::cout << std::endl;
    }
    void print2(std::vector<std::vector<double>> item)
    {
        for (auto i : item)
        {
            for (auto j : i)
                std::cout << j << " ";
            std::cout << std::endl;
        }
    }
    void printMatrix(std::vector<matrix<double>> item)
    {
        for (auto i : item)
        {
            std::cout << i << std::endl;
        }
    }
};

int main()
{

    CSVReader reader("iris.data");
    //     // Get the data from CSV File
    std::vector<std::vector<std::string> > dataList = reader.getData();
    
    // std::vector<double> input = {1, 2};
    // std::vector<int> nodosh{3};
    // std::vector<int> sds{1, 0};
    // // std::cout << "creando objeto mlp\n";

    // MLP mlp = MLP(input, nodosh, sds);
    // mlp.trainning(100, 0.5);
    // mlp.forward();
    // double alpha = 0.05;
    // mlp.backward(alpha);
    return 0;
}
