
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

// 方法2 迭代写法
class Solution {
public:
    vector<vector<int>> permute(vector<int>& nums) {
        vector<vector<int>> res;
        int n = nums.size();
        int len = 1;
        for (int i = n; i >= 1; i--) {
            len *= i;
        }
        res.push_back(nums);
        for (int i = 1; i < len; i++) {
            next(nums);
            res.push_back(nums);
        }

        return res;
    }

    void next(vector<int>& nums) {
        int n = nums.size();
        int k = n - 1;
        while (k >= 1 && nums[k - 1] >= nums[k]) k--;

        if (k == 0) {
            reverse(nums.begin(), nums.end());
            return;
        }

        int j = k - 1;
        while (j + 1 < n && nums[j + 1] > nums[k - 1]) j++;
        swap(nums[j], nums[k - 1]);
        reverse(nums.begin() + k, nums.end());
    }
};