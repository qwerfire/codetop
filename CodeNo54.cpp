class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<bool>> st(n, vector<bool>(m));
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};
        int id = 1;
        vector<int> res;
        int x = 0, y = 0;
        for (int i = 0; i < n * m; i++) {
            res.push_back(matrix[x][y]);
            st[x][y] = true;
            int a = x + dx[id], b = y + dy[id];
            if (a >= 0 && a < n && b >= 0 && b < m && !st[a][b]) {
                x = a, y = b;
            } else {
                id = (id + 1) % 4;
                x = x + dx[id];
                y = y + dy[id];
            }
        }

        return res;
    }
};