#include <iostream>
#include <sstream>
#include <fstream>
#include <iterator>
#include <algorithm>
#include <random>
#include "MLP2.h"
#include "image.h"
#include <filesystem>

using std::filesystem::directory_iterator;

std::vector<std::vector<std::string> > parseCSV()
{
    std::ifstream data("iris.csv");
    std::string line;
    std::vector<std::vector<std::string> > parsedCsv;
    while(std::getline(data,line))
    {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> parsedRow;
        while(std::getline(lineStream,cell,','))
        {
            parsedRow.push_back(cell);
        }

        parsedCsv.push_back(parsedRow);
    }


    return parsedCsv;

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

int main (){

    auto data = parseCSV();
    auto rng = std::default_random_engine {};
  
    // std::vector<std::vector<std::string> > training;
    // std::copy(std::begin(data), std::begin(data) + (data.size() * 0.8), std::back_inserter(training));
    // std::vector<std::vector<std::string> > test;
    // std::copy(std::begin(data) + (data.size() * 0.8), std::end(data), std::back_inserter(test));
    std::vector<Image> training = get_images_vectors_from("feature_vectors/training/");
    std::vector<Image> validation = get_images_vectors_from("feature_vectors/validation/");
    std::vector<Image> testing = get_images_vectors_from("feature_vectors/testing/");
    std::shuffle(std::begin(training), std::end(training), rng);

    // cout<<"Training size: "<<training.size()<<endl;
    // cout<<training[0].get_feature_vector().size()<<endl;

    mat trainX(training.size(), training[0].get_feature_vector().size());
    mat trainY(training.size(), 1);
    for (unsigned i=0; i<training.size(); i++){
        for (unsigned j=0;j<training[i].get_feature_vector().size(); j++){
            trainX(i,j) = training[i].get_feature_vector()[j];
        }
         trainY(i,0) =(double)training[i].get_label_index();
    }

    mat testX(testing.size(), testing[0].get_feature_vector().size());
    mat testY(testing.size(), 1);
    for (unsigned i=0; i<testing.size(); i++){
        for (unsigned j=0;j<testing[i].get_feature_vector().size(); j++){
            testX(i,j) = testing[i].get_feature_vector()[j];
        }
         testY(i,0) =(double)testing[i].get_label_index();
    }

    mat validX(validation.size(), validation[0].get_feature_vector().size());
    mat validY(validation.size(), 1);
    for (unsigned i=0; i<validation.size(); i++){
        for (unsigned j=0;j<validation[i].get_feature_vector().size(); j++){
            validX(i,j) = validation[i].get_feature_vector()[j];
        }
         validY(i,0) =(double)validation[i].get_label_index();
    }

    // cout<<testX.row(0).size()<<endl;
    // cout<<testY.row(0)<<endl;

    vd nodosh{10};
    auto fit = MLP(180,nodosh,10,RELU);
    //0,01

    //
    // fit.trainning<mat>(106,0.001,trainX,trainY);

    //validation
    fit.trainning<mat>(20,0.10,validX,validY);
    // fit.testing<mat>(testX,testY);


}