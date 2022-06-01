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
            init(i, j) = distr(eng);
        }
    }
    return init;
}

class MLP
{
private:
    std::vector<double> input;
    std::vector<int> nodosh;
    int output;
    std::vector<matrix<double>> matrices;
    std::vector<std::vector<double>> deltas;
    std::vector<int> sds;

public:
    MLP(std::vector<double> &input, std::vector<int> &nodosh,int &output,std::vector<int> sds)
    {
        // std::cout << "entrando a constructor\n";
        this->input = input;
        this->nodosh = nodosh;
        this->output = output;
        this->sds = sds;

        // First layer
        matrix<double> init = getMatrix(input.size(), nodosh[0]);
        matrices.push_back(init);
        //deltas por cada nodoh
        
        // std::cout << "inicializando matrices\n";

        // Intermediate layers
        
        for (int i = 1; i < nodosh.size(); ++i)
        {
            // std::cout << "i: " << i << "\n";
            matrix<double> tmp = getMatrix(nodosh[i - 1], nodosh[i]);
            matrices.push_back(tmp);
            
        }

        // std::cout << "intermediate layers done\n";

        // Output Layer
        // std::cout << "prueba " << nodosh[nodosh.size() - 1] << "\n";
        matrix<double> moutput = getMatrix(nodosh[nodosh.size() - 1],this->output);
        matrices.push_back(moutput);
        // std::cout << "constructor ok\n";
    }

    std::vector<double> forward()
    {
        int niveles_h = nodosh.size();
        std::vector<double> hk_prev(input.size());
        for (int i =0; i<  input.size();i++){
             hk_prev[i] = input[i];
        }
        //print(hk_prev);
        for (int i = 0; i < matrices.size(); ++i)
        {
            hk_prev = this->calculate(hk_prev, matrices[i]);
            for(int j=0; j<hk_prev.size();j++){
                hk_prev[j]=logistic(hk_prev[j]);
            }
            this->print(hk_prev);
        }


        return hk_prev;
    }

    double logistic(double x)
    {
        return 1 / (1 + std::exp(-x));
    }

    std::vector<double> calculate(std::vector<double> hk_prev, matrix<double> hk)
    {
        matrix<double> hk_prev_to_matrix(1, hk_prev.size());
        for (int i = 0; i < hk_prev.size(); ++i) {
            hk_prev_to_matrix(0, i) = hk_prev[i];
        }
        matrix<double> result = prod(hk_prev_to_matrix, hk);
        std::vector<double> vector_result(hk.size2(), 0);
        for (int i = 0; i < hk.size2(); ++i) {
            vector_result[i] = result(0, i);
        }
        return vector_result;
    }

    void backward(double alpha)
    {
        //Actualizar Wo
        //necesitamos guardar las salidas por cada capa.
        //por cada item de la ultima matriz 
        //sd -> salida deseada
        //si -> salida 
        // dL/dWij = (Sj - Sdj)Sj(1-Sdj)Sihk 

        auto lastM= matrices[matrices.size()-1];
        for ( auto i=0; i<lastM.size1();i++){
            for (auto j =0; i<lastM.size2();j++){
                //lastM(i,j)= lastM(i,j) - alpha*();
            }
        }
        
    }

    void print(std::vector<double> item){
        for(auto i: item) std::cout<< i <<" ";
        std::cout<<std::endl;
    }
    
};

int main()
{
    std::vector<double> input{1, 2, 3, 4};
    std::vector<int> nodosh{2, 3};
    std::vector<int> sds{1,4};
    int output = 3;

    // std::cout << "creando objeto mlp\n";

    MLP mlp = MLP(input, nodosh, output,sds);
    auto last =mlp.forward();
    return 0;
}
