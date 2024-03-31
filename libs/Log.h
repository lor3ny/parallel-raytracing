#pragma once

#include <iostream>
#include <string>


class Log {

    public:

    static const void Print(const std::string text){
        std::cout << text << std::endl;
    }

     static const void Print(const int text){
        std::cout << text << std::endl;
    }

    static const void PrintError(const std::string text){
        std::cerr << text << std::endl;
    }

    static const void PrintMatrix(const int* buff, int width, int height){
        for(int i = 0; i<width; i++){
            for(int j = 0; j<height; j++){
                std::cout << buff[i*height+j];
            }
            std::cout << std::endl;
        }
    }

};