//方法1
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_set<char> h;
        int res = 0;

        for (int i = 0; i < s.size(); i++) {
            int j = i;
            while (j < s.size() && h.count(s[j]) == 0) {
                h.insert(s[j]);
                j++;
            }

            res = max(res, j - i);
            h.clear();
        }

        return res;
    }
};

// 方法2
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char, int> h;
        int res = 0;

        for (int i = 0, j = 0; i < s.size(); i++) {
            h[s[i]]++;
            while (h[s[i]] > 1) h[s[j++]]--;
            res = max(res, i - j + 1);
    
        }

        return res;
    }
};
