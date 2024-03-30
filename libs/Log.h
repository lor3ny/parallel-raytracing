#pragma once

#include <iostream>
#include <string>


class Log {

    public:

    static const void Print(const std::string text){
        std::cout << text << std::endl;
    }

    static const void PrintError(const std::string text){
        std::cerr << text << std::endl;
    }

};