
// 方法1 经典做法
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;
    vector<bool> st;
    int n;

    vector<vector<int>> permute(vector<int>& nums) {
        n = nums.size();
        st = vector<bool>(n);
        dfs(nums);

        return res;
    }

    void dfs(vector<int>& nums) {
        if (path.size() == n) {
            res.push_back(path);
            return;
        }

        for (int i = 0; i < n; i++) {
            if (!st[i]) {
                path.push_back(nums[i]);
                st[i] = true;
                dfs(nums);
                path.pop_back();
                st[i] = false;
            }
        }
    }
};
