# 54. 螺旋矩阵
```
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
```


# 23. 合并K个升序链表
### 方法1
```
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        auto dummy = new ListNode();
        ListNode* p = nullptr;
        for (auto l : lists) {
            auto t = mergeTwoLists(p, l);
            dummy->next = t;
            p = dummy->next;
        }

        return dummy->next;
    }

    ListNode* mergeTwoLists(ListNode* h1, ListNode* h2) {
        auto dummy = new ListNode();
        auto p = dummy;
        while (h1 && h2) {
            if (h1->val <= h2->val) {
                p->next = h1;
                h1 = h1->next;
            }  else {
                p->next = h2;
                h2 = h2->next;
            }
            p = p->next;
        }

        if (h1) p->next = h1;
        if (h2) p->next = h2;

        return dummy->next;
    }
};
```

### 方法2 LogN的做法
```
class Solution {
public:
    ListNode* mergeKLists(vector<ListNode*>& lists) {
        int n = lists.size();
        if (n == 0) return nullptr;
        int s = 1;
        while (s < n)
        {
            for (int i = 0; i + s < n; i += 2 * s) {
                lists[i] = mergeTwoLists(lists[i], lists[i + s]);
            }
            s <<= 1;
        }

        return lists[0];
    }

    ListNode* mergeTwoLists(ListNode* l1, ListNode* l2) {
        if (l1 == nullptr) return l2;
        if (l2 == nullptr) return l1;
        auto dummy = new ListNode(0);
        auto cur = dummy;
        while (l1 && l2) {
            if (l1->val < l2->val) {
                cur->next = l1;
                l1 = l1->next;
            } else {
                cur->next = l2;
                l2 = l2->next;
            }
            cur = cur->next;
        }

        if (l1) {
            cur->next = l1;
        }

        if (l2) {
            cur->next = l2;
        }

        return dummy->next;
    }
};
```
### 方法3 优先队列做法，注意小根堆的库实现
```
class Solution {
public:
    struct Cmp {
        bool operator()(ListNode* a, ListNode* b) {
            return a->val > b->val; // 小根堆 大于号； 大根堆 小于号
        }
    };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*, vector<ListNode*>, Cmp> heap;
        for (auto l: lists) {
            if (l) //这句不能少
            {
                heap.push(l);
            }
        }
        auto dummy = new ListNode(-1);
        auto cur = dummy;
        while (heap.size()) {
            auto t = heap.top();
            heap.pop();
            cur = cur->next = t;
            // cur = cur->next;
            if (t->next) {
                heap.push(t->next);
            }
        }

        return dummy->next;
    }
};
```
# 415. 字符串相加

```
class Solution {
public:
    string addStrings(string num1, string num2) {
        reverse(num1.begin(), num1.end());
        reverse(num2.begin(), num2.end());

        int c = 0;
        int i = 0, j = 0, l1 = num1.size(), l2 = num2.size();
        string res;
        while (i < l1 && j < l2) {
            c = num1[i] - '0' + num2[j] - '0' + c;
            res += '0' + (c % 10);
            c /= 10;
            i++, j++;
        }
        while (i < l1) {
            c = num1[i] - '0' + c;
            res += '0' + (c % 10);
            c /= 10;
            i++;
        }

        while (j < l2) {
            c = num2[j] - '0' + c;
            res += '0' + (c % 10);
            c /= 10;
            j++;
        }

        if (c) res += '0' + c;
        reverse(res.begin(), res.end());

        return res;
    }
};
```

# 142. 环形链表 II
### 方法1 快慢指针
```
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        auto f = head, s = head;
        while (1) {
            if (f && f->next) {
                f = f->next->next;
            } else {
                return NULL;
            }
            s = s->next;
            if (f == s) break;
        }

        f = head;
        while (f != s) {
            f = f->next;
            s = s->next;
        }

        return f;
    }
};
```
# 300. 最长上升子序列
### 方法1 dp
```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1);

        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = i; j < n; j++) {
                if (nums[j] > nums[i]) {
                    dp[j] = max(dp[j], dp[i] + 1);
                }
                res = max(res, dp[i]);
            }
        }

        return res;
    }
};
```
### 方法2 二分
```
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        int n = nums.size();
        int len = 0;
        if (n <= 1) return n;

        // [4,10,4,3,8,9]

        // help就是最后的上升子序列，不一定是
        vector<int> help(n);
        help[0] = nums[0];
        for (int i = 1; i < n; i++) {
            int x = nums[i];
            int l = 0, r = len;
            while (l < r) {
                int mid = l + r >> 1;
                if (help[mid] >= x) r = mid;
                else l = mid + 1;
            }
            /* for debug
            cout << "x = " << x << ", l = " << l << endl;
            for (int j = 0; j <= len; j++) {
                cout << help[j] << " ";
            }
            cout << endl;
            */

            if (help[l] >= x) {
                help[l] = x;
            } else {
                help[++len] = x;
            }
        }

        // for (int j = 0; j <= len; j++) {
        //     cout << help[j] << " ";
        // }
        return len + 1;
    }
};
```

# 42. 接雨水
### 方法1 双指针，按列累加
```
class Solution {
public:
    int trap(vector<int>& h) {
        int n = h.size();
        vector<int> l(n), r(n);
        for (int i = 1; i < n; i++) {
            l[i] = max(h[i - 1], l[i - 1]);
        }

        for (int i = n - 2; i >= 0; i--) {
            r[i] = max(h[i + 1], r[i + 1]);
        }

        int res = 0;
        for (int i = 0; i < n; i++) {
            int t = min(l[i], r[i]);
            if (t > h[i]) res += t - h[i];
        }

        return res;
    }
};
```

### 方法2 单调栈
```
class Solution {
public:
    int trap(vector<int>& height) {
        stack<int> st;
        int res = 0;

        for (int i = 0; i < height.size(); i++) {
            while (st.size() && height[st.top()] < height[i]) {
                int t = st.top();
                st.pop();
                if (st.empty()) {
                    break;
                }
                int w = min(height[i], height[st.top()]) - height[t];
                int l = i - st.top() - 1;
                res += w * l;
            }
            st.push(i);
        }

        return res;
    }
};
```
# Leetcode 124. 二叉树中的最大路径和
```
class Solution {
public:
    int res;
    int maxPathSum(TreeNode* root) {
        res = INT_MIN;
        dfs(root);

        return res;
    }

    int dfs(TreeNode* root)
    {
        if (!root) return 0;
        int l = max(0, dfs(root->left));
        int r = max(0, dfs(root->right));

        res = max(res, l + r + root->val);

        return max(l, r) +  root->val;
    }
};
```
# 143. 重排链表
### 方法1
```
class Solution {
public:
    int getNo(ListNode* p) {
        int res = 0;
        while (p) {
            res++;
            p = p->next;
        }

        return res;
    }
    void reorderList(ListNode* head) {
        int n = getNo(head);
        if (n <= 2) return;
        ListNode* pre = head, *mid = head;
        for (int i = 0; i < n / 2; i++) {
            pre = mid;
            mid = mid->next;
        }
        pre->next = nullptr;
        auto l = head;
        auto r = reverse(mid);

        while (l && l->next) {
            auto tl = l->next;
            auto tr = r->next;
            l->next = r;
            r->next = tl;
            l = tl;
            r = tr;
        }
        l->next = r;
    }

    ListNode* reverse(ListNode* node)
    {
        if (!node || !node->next) return node;
        ListNode* a = node, *b = node->next;
        while (b) {
            auto c = b->next;
            b->next = a;
            a = b;
            b = c;
        }
        node->next = nullptr;
        return a;
    }
};
```

### 方法2
```
class Solution {
public:
    int getNo(ListNode* p) {
        int res = 0;
        while (p) {
            res++;
            p = p->next;
        }

        return res;
    }
    void reorderList(ListNode* head) {
        int n = getNo(head);
        if (n <= 2) return;
        ListNode* pre = head, *mid = head;
        for (int i = 0; i < n / 2; i++) {
            pre = mid;
            mid = mid->next;
        }
        pre->next = nullptr;
        auto l = head;
        auto r = reverse(mid);

        // while (l && l->next) {
        //     auto tl = l->next;
        //     auto tr = r->next;
        //     l->next = r;
        //     r->next = tl;
        //     l = tl;
        //     r = tr;
        // }
        // l->next = r;

        auto p = new ListNode();
        int cnt = 0;
        while (l && r) {
            if (cnt == 0) {
                p->next = l;
                l = l->next;
                p = p->next;
                cnt = 1;
            } else {
                cnt = 0;
                p->next = r;
                r = r->next;
                p = p->next;
            }
        }
        p->next = r;
    }

    ListNode* reverse(ListNode* node)
    {
        if (!node || !node->next) return node;
        ListNode* a = node, *b = node->next;
        while (b) {
            auto c = b->next;
            b->next = a;
            a = b;
            b = c;
        }
        node->next = nullptr;
        return a;
    }
};
```
# 94. 二叉树的中序遍历
### 方法1 递归
```
class Solution {
public:
    vector<int> ans;

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        ans.push_back(root->val);
        dfs(root->right);
    }
    vector<int> inorderTraversal(TreeNode* root) {
        dfs(root);
        return ans;
    }
};
```

### 方法2 迭代写法
```
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        stack<TreeNode*> st;

        vector<int> res;
        while (st.size() || root) {
            if (root) {
                st.push(root);
                root = root->left;
            } else {
                auto t = st.top();
                st.pop();
                res.push_back(t->val);
                root = t->right;
            }
        }

        return res;
    }
};
```

# 199. 二叉树的右视图
### 方法1 bfs
```

class Solution {
public:
    vector<int> res;
    void dfs(TreeNode* root, int d)
    {
        if (!root) return;
        if (res.size() == d) res.push_back(root->val);
        dfs(root->right, d + 1);
        dfs(root->left, d + 1);
    }

    vector<int> rightSideView(TreeNode* root) {
        dfs(root, 0);

        return res;
    }
};
```


### 方法2 dfs

```
class Solution {
public:
    vector<int> res;
    void dfs(TreeNode* root, int d)
    {
        if (!root) return;
        if (res.size() == d) res.push_back(root->val);
        dfs(root->right, d + 1);
        dfs(root->left, d + 1);
    }

    vector<int> rightSideView(TreeNode* root) {
        dfs(root, 0);

        return res;
    }
};
```
# 19. 删除链表的倒数第 N 个结点
```
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int n) {
        auto dummy = new ListNode(-1);
        dummy->next = head;
        auto slow = dummy, fast = head;
        for (int i = 1; i <= n; i++) {
            fast = fast->next;
        }
        while (fast) {
            slow = slow->next;
            fast = fast->next;
        }

        auto cur = slow->next;;
        auto post = cur->next;
        slow->next = post;

        return dummy->next;
    }
};
```
# 232. 用栈实现队列
```
class MyQueue {
public:
    stack<int> st1, st2;

    MyQueue() {

    }
    
    void push(int x) {
        st1.push(x);
    }
    
    int pop() {
        while (st1.size()) {
            st2.push(st1.top());
            st1.pop();
        }
        int res = st2.top();
        st2.pop();

        while (st2.size()) {
            st1.push(st2.top());
            st2.pop();
        }

        return res;
    }
    
    int peek() {
        while (st1.size()) {
            st2.push(st1.top());
            st1.pop();
        }
        int res = st2.top();

        while (st2.size()) {
            st1.push(st2.top());
            st2.pop();
        }

        return res;
    }
    
    bool empty() {
        return st1.empty();
    }
};

/**
 * Your MyQueue object will be instantiated and called as such:
 * MyQueue* obj = new MyQueue();
 * obj->push(x);
 * int param_2 = obj->pop();
 * int param_3 = obj->peek();
 * bool param_4 = obj->empty();
 */
 ```