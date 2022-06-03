#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <bits/stdc++.h>
#include <iomanip>

// forward: input * primera matriz y backward(input)
using namespace boost::numeric::ublas;

matrix<double> getMatrix(int const &nRows, int const &nCols)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0, 10);

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
    std::vector<int> sds;

public:
    MLP(std::vector<double> &input, std::vector<int> &nodosh, int &output, std::vector<int> sds)
    {
        // this->input;
        this->input = vector<double>(input.size());
        std::copy(input.begin(), input.end(), this->input.begin());
        this->nodosh = nodosh;
        this->output = output;
        this->sds = sds;

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
                ss[j] = hk_prev[j] = sigmoide(hk_prev[j]);
            }

            // if (matrices.size() - 1 == i)
            // {
            //     for (int j = 0; j < ss.size(); j++)
            //     {
            //         ss[j] = (ss[j] - sds[j]) * ss[j] * (1 - sds[j]);
            //     }
            // }
            soutouts.push_back(ss);
            std::cout << "hk_final" << i << ": " << hk_prev << std::endl;
        }

        // print2(deltas);

        return hk_prev;
    }

    double sigmoide(double x)
    {
        return 1 / (1 + std::exp(-x));
    }
    void backward(double alpha)
    {
        // Actualizar Wo
        // necesitamos guardar las salidas por cada capa.
        // por cada item de la ultima matriz
        // sd -> salida deseada
        // si -> salida
        //  dL/dWij = (Sj - Sdj)Sj(1-Sdj)Sihk
        auto matricres_cp = matrices;
        std::vector<double> curDeltas;

        for (unsigned i=matrices.size()-1; i>=0; i--){
                auto currentM = matrices[i];
                auto changeM = matricres_cp[i];
                auto curOuputs = soutouts[i];

                if(matrices.size()-1==i){
                    // hk con ouput
                    auto outputsK = soutouts[i-1];
                    //precalculo el delta
                    for(unsigned j=0; j<curOuputs.size(); j++){
                        curDeltas.push_back( (curOuputs[j]- sds[j]) * sds[j] *(1-sds[j]) );
                    }
                    for (int j = 0; j < currentM.size1(); j++)
                    {
                        for (int k = 0; k < currentM.size2(); k++)
                        {
                            changeM(j, k) = currentM(j, k) - alpha * (curDeltas[i] * outputsK[j]);
                        }
                    }
                }
                else if(i==0){
                    // input con h1
                    auto outpusI = input;
                    std::vector<double> tmpdelta;
                    //preproserar el delta
                    for( int j =0; j<currentM.size1(); j++){
                        for (int k = 0; k< currentM.size2(); k++){
                             
                        }
                    }



                }
                else {
                    // hk-1 con hk;
                }


        }

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
};

int main()
{
    std::vector<double> input = {1, 2};
    std::vector<int> nodosh{3};
    std::vector<int> sds{2, 4};
    int output = 2;
    // std::cout << "creando objeto mlp\n";

    MLP mlp = MLP(input, nodosh, output, sds);
    auto last = mlp.forward();
    double alpha = 0.05;
    mlp.backward(alpha);

    return 0;
}
