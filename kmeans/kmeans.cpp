#include <bits/stdc++.h>

using namespace std;

class Solution
{
  public:
    Solution(vector<vector<float>> b, int c)
    {
        X = b;
        K = c;
    }

    vector<int> assignLabels(vector<vector<float>> &X,
                             vector<vector<float>> &centers)
    {
        int N = X.size();
        int D = X[0].size();
        int K = centers.size();
        vector<int> labels(N);

        for (int i = 0; i < N; i++)
        {
            float dist = numeric_limits<float>::max();
            for (int j = 0; j < K; j++)
            {
                float curr_dist = 0;
                for (int k = 0; k < D; k++)
                {
                    curr_dist += pow((X[i][k] - centers[j][k]), 2);
                }
                if (curr_dist < dist)
                {
                    dist = curr_dist;
                    labels[i] = j;
                }
            }
        }

        return labels;
    }

    vector<vector<float>> kmeans()
    {
        // X is N x D table, return cluster centers (KxD) and labels (N)
        int N = X.size();
        int D = X[0].size();
        vector<vector<float>> centers(K, vector<float>(D, 0));
        for (int i = 0; i < K; i++)
        {
            centers[i] = X[i];
        }
        vector<int> labels = assignLabels(X, centers);

        int num_iters = 0;
        while (true)
        {
            vector<int> numDatainCenter(K, 0);

            // Get cluster centers
            for (int i = 0; i < N; i++)
            {
                int center_label = labels[i];
                for (int j = 0; j < D; j++)
                {
                    centers[center_label][j] += X[i][j];
                }
                numDatainCenter[center_label] += 1;
            }

            for (int i = 0; i < K; i++)
            {
                for (int j = 0; j < D; j++)
                {
                    centers[i][j] /= numDatainCenter[i];
                }
            }

            // Assign labels according to these centers
            vector<int> new_labels = assignLabels(X, centers);
            bool converge = true;

            for (int i = 0; i < N; i++)
            {
                if (new_labels[i] != labels[i])
                {
                    converge = false;
                    break;
                }
            }

            if (converge)
            {
                break;
            }

            num_iters++;
            labels = new_labels;
        }
        cout << num_iters << endl;
        return centers;
    }

  private:
    vector<vector<float>> X;
    int K;
};

int main()
{
    vector<vector<string>> content;
    vector<string> row;
    string line, word;
    string fname = "adult.csv";

    fstream file(fname, ios::in);
    if (file.is_open())
    {
        while (getline(file, line))
        {
            row.clear();

            stringstream str(line);

            while (getline(str, word, ','))
                row.push_back(word);
            content.push_back(row);
        }
    }

    vector<vector<float>> X;
    int data_size = content.size();
    int N = min(30000, data_size);

    for (int i = 0; i < N; i++)
    {
        vector<float> x_row;
        for (int j = 0; j < content[i].size(); j++)
        {
            if (j == 0 || j == 2 || j == 4)
            {
                x_row.push_back(stoi(content[i][j]));
            }
        }
        X.push_back(x_row);
    }

    int D = X[0].size();
    int K = 10;

    cout << N << " " << D << " " << K << endl;

    vector<float> mean(D, 0);
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < N; j++)
        {
            mean[i] += X[j][i];
        }
        mean[i] /= float(N);
    }

    vector<float> stdev(D, 0);
    for (int i = 0; i < D; i++)
    {
        for (int j = 0; j < N; j++)
        {
            stdev[i] += (X[j][i] - mean[i]) * (X[j][i] - mean[i]);
        }
        stdev[i] /= float(N);
        stdev[i] = sqrt(stdev[i]);
    }

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < D; j++)
        {
            X[i][j] = (X[i][j] - mean[j]) / stdev[j];
        }
    }

    Solution solution(X, K);
    vector<vector<float>> centers = solution.kmeans();
    for (int i = 0; i < K; i++)
    {
        for (int j = 0; j < D; j++)
        {
            cout << centers[i][j] << " ";
        }
        cout << endl;
    }

    // vector<int> labels = solution.assignLabels(X, centers);
    // for (int i = 0; i < labels.size(); i++)
    // {
    //     cout << labels[i] << " ";
    // }

    return 0;
}
