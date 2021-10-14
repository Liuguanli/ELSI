#ifndef UTIL_H
#define UTIL_H

#include <string>
#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <iomanip>
#include <sstream>

using namespace std;

namespace file_utils
{
    inline int check_dir(string path);
};

namespace file_utils
{
    int check_dir(string path)
    {
        std::ifstream fin(path);
        if (!fin)
        {
            string command = "mkdir -p " + path;
            return system(command.c_str());
        }
        return 0;
    }
}

// namespace string_utils
// {
//     vector<string> split(const string &str, const string &pattern)
//     {
//         vector<string> res;
//         if (str == "")
//             return res;
//         string strs = str + pattern;
//         size_t pos = strs.find(pattern);

//         while (pos != strs.npos)
//         {
//             string temp = strs.substr(0, pos);
//             res.push_back(temp);
//             strs = strs.substr(pos + 1, strs.size());
//             pos = strs.find(pattern);
//         }

//         return res;
//     }
// }

#endif