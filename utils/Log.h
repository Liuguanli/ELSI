#ifndef LOG_H
#define LOG_H

#include <iostream>
#include <fstream>
#include <boost/algorithm/string.hpp>
#include <vector>
#include <iterator>
#include <string>
#include <algorithm>
#include <random>
#include <math.h>
#include <cmath>
#include <iomanip>
#include <sstream>
#include <cstdlib>
#include <ctime>

namespace logger
{
    bool is_closed = false;

    void print(string s)
    {
        if (is_closed)
        {
            return;
        }
        cout << s << endl;
    }
}

#endif