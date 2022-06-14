#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
// #include <boost/filesystem.hpp>
// #include <boost/range/iterator_range.hpp>
#include <bits/stdc++.h>
#include <iomanip>
#include <algorithm>
#include <iostream>
#include "excel.h"
#include "image.h"
#include <random>
#include <filesystem>
#include <cstdlib>

// forward: input * primera matriz y backward(input)
using namespace boost::numeric::ublas;
using std::filesystem::directory_iterator;

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
    std::uniform_real_distribution<double> distr(0, 1);

    matrix<double> init(nRows, nCols);
    for (size_t i = 0; i < nRows; i++)
    {
        for (size_t j = 0; j < nCols; j++)
        {
            init(i, j) = distr(eng);
            // init(i, j) = 1;j
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
        hk_prev = softmax(hk_prev);
        std::vector<double> stmp(hk_prev.size());
        std::copy(hk_prev.begin(), hk_prev.end(), stmp.begin());
        soutouts.push_back(stmp);
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
            std::cout << "eOuput: " << eOuput << std::endl;
            // calcular error y guardar el error;
            auto currentError = this->CEE(eOuput);
            std::cout << currentError << std::endl;
            // backwards
            // std::cout << "AcualW:";
            // printMatrix(this->matrices);
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
                curOuputs = soutouts[i + 1];
                outputsK.clear();
                outputsK.resize(tmpL.size());
                std::copy(tmpL.begin(), tmpL.end(), outputsK.begin());
                // precalculo de los deltas
                for (unsigned j = 0; j < output; j++)
                {
                    // auto deltai = (curOuputs[j] - sds[j]) * curOuputs[j] * (1 - curOuputs[j])
                    auto deltai = curOuputs[j] - sds[j];
                    curDeltas.push_back(deltai);
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

    vector<double> softmax(vector<double> x)
    {
        vector<double> result(x.size());
        double sum = 0.0;
        for (unsigned i = 0; i < x.size(); i++)
        {
            sum += exp(x[i]);
        }

        for (unsigned i = 0; i < x.size(); i++)
        {
            result[i] = (exp(x[i]) / sum);
        }
        return result;
    }

    double CEE(vector<double> output)
    {
        double vError = 0;
        for (unsigned i = 0; i < output.size(); i++)
        {
            vError += (sds[i] * log(output[i]));
        }
        return vError * -1.0;
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

std::vector<Image> get_images_vectors_from(std::string folder_path)
{
    std::vector<Image> images;
    int i = 0;
    for (const auto &file : directory_iterator(folder_path))
    {
        images.push_back(Image(file.path()));
    }
    return images;
}

int main()
{

    // CSVReader reader("iris.data");
    // //     // Get the data from CSV File
    // std::vector<std::vector<std::string>> dataList = reader.getData();

    std::vector<Image> training = get_images_vectors_from("feature_vectors/training/");
    std::vector<Image> validation = get_images_vectors_from("feature_vectors/validation/");
    std::vector<Image> testing = get_images_vectors_from("feature_vectors/testing/");

    auto input = training[0].get_feature_vector();
    auto itemLabel = training[0].get_label();
    std::vector<int> sds;
    for (auto i : itemLabel)
    {
        if (i == '0')
        {
            sds.push_back(0);
        }
        else
        {
            sds.push_back(1);
        }

        // double num_double = std::atof(i);
    }

    // std::cout<<std::endl;
    // std::cout<<itemLabel<<std::endl;

    // std::vector<double> input = {0.84, 0.33};
    std::vector<int> nodosh{100, 50, 25, 10};
    // std::vector<int> sds{1, 0};
    // // 0.5,0     -(1log(0.5) + 1log(0))
    // // std::cout << "creando objeto mlp\n";

    MLP mlp = MLP(input, nodosh, sds);
    mlp.trainning(100, 0.06);
    return 0;
}
