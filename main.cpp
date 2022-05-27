#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <bits/stdc++.h>
#include <iomanip>

// forward: input * primera matriz y backward(input)
using namespace boost::numeric::ublas;

matrix<double> getMatrix(int nRows, int nCols)
{
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<double> distr(0, 10);

    matrix<double> init(nRows, nCols);
    for(size_t i=0; i<init.size1(); i++) {
        for(size_t j=0; j<init.size2(); j++) {
            init(i,j) = distr(eng);
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
    
public:
    MLP(std::vector<double> input, std::vector<int> nodosh) {

        this->input = input;
        this->nodosh = nodosh;

        // First layer 
        matrix<double> init = getMatrix(input.size(), nodosh[0]);
        matrices.push_back(init);

        // Intermediate layers
        for (int i = 1; i < nodosh.size()-1; ++i)
        {
            matrix<double> tmp = getMatrix(nodosh[i - 1], nodosh[i]);
            matrices.push_back(tmp);
        }

        // Output Layer
        matrix<double> moutput = getMatrix(nodosh[nodosh.size()-1],output);
        matrices.push_back(moutput);
        
    }

    void calculate() {
        int niveles_h = nodosh.size();
        matrix<double> hk_prev(1, input.size());
        //matrices.push_back(hk_prev);
        // ingresar a hk_prev el input.
        for (int i = 0; i <= niveles_h; ++i)
            hk_prev(0, i) = input[i];
            //neta por nivel
            matrix<double> hk_prev = this->forward(hk_prev, matrices[i]);
            //funcion de activcacion.
            std::transform(hk_prev.begin1(),hk_prev.begin2(),logistic);
            //se actualiza el input.
            hk = matrices[i]
            // matrices[]
            // hk_prev = hk;
            // hk = result;
        }
    }


    double logistic(double x){
        return 1 / (1 + exp(-x));
    }
    
    
    matrix<double> forward(matrix<double> hk_prev, matrix<double> hk)
    {
        return prod(hk_prev, hk);
    }

    void backward() {

    }

}


// void net(std::vector<int> input, std::vector<int> nodosh, int output){
//     // 1 * len(input) para los x's del input
//     // len(input) * nodosh[0]
//     // net_1 = I * W^(Ih) + b^(Ih)
// }

int main()
{
   
    // matrix<double> m(3, 3);
    //  std::vector<matrix<double>> m;
    // vector<matrix<double>> m3;
    // matrix<double> m2(1, 3);
    // for (unsigned i = 0; i < m.size1(); ++i)
    //     for (unsigned j = 0; j < m.size2(); ++j)
    //         m(i, j) = 3 * i + j;

    // for (unsigned i = 0; i < m2.size1(); ++i)
    //     for (unsigned j = 0; j < m2.size2(); ++j)
    //         m2(i, j) = 3 * i + j;

    // std::cout << m << std::endl;
    // std::cout << m2 << std::endl;
    // matrix<double> C;
    // C = prod(m2,m);
    // std::cout<<C;
    return 0;
}
