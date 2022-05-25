#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <bits/stdc++.h>

// forward: input * primera matriz y backward(input)
using namespace boost::numeric::ublas;

class MLP
{
private:
    std::vector<double> input;
    std::vector<int> nodosh;
    std::vector<matrix<double> > matrices;
    std::vector<vector<double>> weights;
    
public:
    MLP(std::vector<double> input, std::vector<int> nodosh) {
        this->input = input;
        this->nodosh = nodosh;
    }

    void calculate() {
        int niveles_h = nodosh.size();
        matrix<double> hk_prev(1, input.size()), hk(input.size(), nodosh[0]);
        matrices.push_back(hk_prev);
        // TODO: ingresar a hk_prev el input.
        for (int i = 0; i < input.size(); ++i)
            hk_prev(0, i) = input[i];
        // TODO: ingresar los nodos del nivel h 1 a hk

        for (int i = 0; i < niveles_h; ++i) {
            matrix<double> result = this->forward(hk_prev, hk);
            matrices.push_back(hk);
            hk_prev = hk;
            hk = result;
        }
    }
    
    void forward(matrix<double> hk_prev, matrix<double> hk)
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
