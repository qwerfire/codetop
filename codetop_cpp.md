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
# 144. 二叉树的前序遍历
### 方法1 递归
 ```
class Solution {
public:
    vector<int> res;
    vector<int> preorderTraversal(TreeNode* root) {
        dfs(root);

        return res;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        res.push_back(root->val);
        dfs(root->left);
        dfs(root->right);
    }
};
```

### 方法2 迭代
```
class Solution {
public:
    vector<int> preorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> st;
        while (st.size() || root) {
            if (root) {
                res.push_back(root->val);
                st.push(root);
                root = root->left;
            } else {
                root = st.top()->right;
                st.pop();
            }
        }

        return res;
    }
};
```
# 4. 寻找两个正序数组的中位数
```
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int cnt = nums1.size() + nums2.size();
        if (cnt % 2) return findk(nums1, 0, nums2, 0, cnt / 2 + 1);

        return (findk(nums1, 0, nums2, 0, cnt / 2) + findk(nums1, 0, nums2, 0, cnt / 2 + 1)) / 2.0;
    }

    int findk(vector<int>& nums1, int i, vector<int>& nums2, int j, int k)
    {
        if (nums1.size() - i > nums2.size() - j) return findk(nums2, j, nums1, i, k);

        if (nums1.size() == i) return nums2[j + k - 1];// 这句必须在前面，否则下一句i可能越界
        if (k == 1) return min(nums1[i], nums2[j]);

        int si = min((int)nums1.size(), i + k / 2); 
        int sj = j + k / 2;
        if (nums1[si - 1] > nums2[sj - 1]) {
            return findk(nums1, i, nums2, sj, k - (sj - j));
        } else {
            return findk(nums1, si, nums2, j, k - (si - i));
        }
    }
};
```
# 69. x 的平方根 
### 二分
```
class Solution {
public:
    int mySqrt(int x) {
        int l = 0, r = x;
        while (l < r) {
            int mid = l + r + 1ll >> 1;
            if (mid > x / mid) r = mid - 1;
            else l = mid;
        }

        return l;
    }
};
```
# 72. 编辑距离
```
// 相似题目
https://www.nowcoder.com/practice/05fed41805ae4394ab6607d0d745c8e4?tpId=196&&tqId=37134&rp=1&ru=/ta/job-code-total&qru=/ta/job-code-total/question-ranking
```
```
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        for (int i = 0; i <= n ;i++) {
            dp[i][0] = i;
        }

        for (int i = 0; i <= m ;i++) {
            dp[0][i] = i;
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1];
                } else {
                    dp[i][j] = min(dp[i - 1][j - 1], min(dp[i][j - 1], dp[i - 1][j])) + 1;
                }
            }
        }

        return dp[n][m];
    }
};
```
# 56. 合并区间
```
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(), [](vector<int> a, vector<int> b){
            if (a[0] == b[0]) return a[1] < b[1];
            return a[0] < b[0];
        });

        vector<vector<int>> res;
        int l = intervals[0][0], r = intervals[0][1];
        for (int i = 1; i < intervals.size(); i++) {
            int nl = intervals[i][0], nr = intervals[i][1];
            if (nl > r) {
                res.push_back({l, r});
                l = nl, r = nr;
            } else {
                r = max(r, nr);
            }
        }

        res.push_back({l, r});

        // for (auto d : intervals) {
        //     cout << d[0] << " " << d[1] << endl;
        // }

        return res;
    }
};
```
# 2. 两数相加
```
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        auto dummy = new ListNode();
        auto p = dummy;
        int c = 0;
        while (l1 && l2) {
            int v = l1->val + l2->val + c;
            auto node = new ListNode(v % 10);
            p->next = node;
            p = p->next;
            c = v / 10; 
            l1 = l1->next;
            l2 = l2->next;
        }

        while (l1) {
            int v = l1->val + c;
            auto node = new ListNode(v % 10);
            p->next = node;
            p = p->next;
            c = v / 10; 
            l1 = l1->next;
        }

        while (l2) {
            int v = l2->val + c;
            auto node = new ListNode(v % 10);
            p->next = node;
            p = p->next;
            c = v / 10; 
            l2 = l2->next;
        }

        if (c) {
            auto node = new ListNode(c);
            p->next = node;
            p = p->next;
        }

        return dummy->next;

    }
};
```
# 148. 排序链表
## *递归方法未实现*
### 方法1 归并
```
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        ListNode* mid = split(head);

        auto l = sortList(head);
        auto r = sortList(mid);

        return merge(l, r);
    }

    ListNode* split(ListNode* node)
    {
        int n = 0;
        ListNode* p = node;
        while (p) {
            n++;
            p = p->next;
        }

        p = node;
        for (int i = 1; i < n / 2; i++) {
            p = p->next;
        }
        auto r = p->next;
        p->next = nullptr;

        return r;
    }

    ListNode* merge(ListNode* l, ListNode* r) 
    {
        auto dummy = new ListNode();
        auto p = dummy;
        while (l && r)
        {
            if (l->val < r->val) {
                p->next = l;
                l = l->next;
                p = p->next;
            } else {
                p->next = r;
                r = r->next;
                p = p->next;
            }
        }

        if (l) {
            p->next = l;
        }

        if (r) {
            p->next = r;
        }

        return dummy->next;
    }
};
```
### 方法2 归并，找中间节点方法修改
```
class Solution {
public:
    ListNode* sortList(ListNode* head) {
        if (head == nullptr || head->next == nullptr) return head;

        ListNode* head1 = head;
        ListNode* head2 = split(head);

        head1 = sortList(head1);        //一条链表分成两段分别递归排序
        head2 = sortList(head2);

        return merge(head1, head2);     //返回合并后结果
    }

    //双指针找单链表中点模板
    ListNode* split(ListNode* head)     
    {
        ListNode *slow = head, *fast = head->next;

        while (fast != nullptr && fast->next != nullptr)
        {
            slow = slow->next;
            fast = fast->next->next;
        }

        ListNode* mid = slow->next;
        slow->next = nullptr;           //断尾

        return mid;
    }

    //合并两个排序链表模板
    ListNode* merge(ListNode* head1, ListNode* head2)
    {
        ListNode *dummy = new ListNode(0), *p = dummy;

        while (head1 != nullptr && head2 != nullptr)
        {
            if (head1->val < head2->val)
            {
                p = p->next = head1;
                head1 = head1->next;
            }
            else
            {
                p = p->next = head2;
                head2 = head2->next;
            }
        }

        if (head1 != nullptr) p->next = head1;
        if (head2 != nullptr) p->next = head2;

        return dummy->next;
    }
};
```

### 方法3 快排 会超时
```
class Solution {
public:
    auto getTail(ListNode* p)
    {
        while (p->next) p = p->next;
        
        return p;
    }
    
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        
        auto left = new ListNode(-1), mid = new ListNode(-1), right = new ListNode(-1);
        auto l = left, m = mid, r = right;
        auto val = head->val;
        for (auto p = head; p; p = p->next) {
            if (p->val < val) l = l->next = p;
            else if (p->val == val) m = m->next = p;
            else r = r->next = p;
        }
        
        l->next = m->next = r->next = nullptr;
        left->next = sortList(left->next);
        right->next = sortList(right->next);
        getTail(left)->next = mid->next;
        getTail(left)->next = right->next;
        
        return left->next;
    }
};
```
### 方法4 快排改进不超时
```
class Solution {
public:
    auto getTail(ListNode* p)
    {
        while (p->next) p = p->next;
        
        return p;
    }

    int getLen(ListNode* p) {
        int n = 0;
        while (p) {
            p = p->next;
            n++;
        }

        return n;
    }

    ListNode* getMid(ListNode* p) {
        int n = getLen(p);
        ListNode* node = p;
        for (int i = 1; i < n / 2; i++) {
            node = node->next;
        }

        return node;
    }
    
    ListNode* sortList(ListNode* head) {
        if (!head || !head->next) return head;
        
        auto left = new ListNode(-1), mid = new ListNode(-1), right = new ListNode(-1);
        auto l = left, m = mid, r = right;
        auto val = getMid(head)->val;
        for (auto p = head; p; p = p->next) {
            if (p->val < val) l = l->next = p;
            else if (p->val == val) m = m->next = p;
            else r = r->next = p;
        }
        
        l->next = m->next = r->next = nullptr;
        left->next = sortList(left->next);
        right->next = sortList(right->next);
        getTail(left)->next = mid->next;
        getTail(left)->next = right->next;
        
        return left->next;
    }
};
```

# 82. 删除排序链表中的重复元素 II
### 方法1
```
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if (!head || !head->next) return head;

        auto p0 = new ListNode(-1000), p1 = new ListNode(-500);
        p0->next = p1;
        p1->next = head;
        auto d0 = p0, d1 = p1;
        while (head) {
            if (p1->val != head->val) {
                p0 = p1;
                p1 = head;
                head = head->next;
            } else {
                while (head && p1->val == head->val)
                    head = head->next;
                p0->next = head;
                p1 = head;
                if (head) head = head->next;
            }
        }

        return d1->next;
    }
};
```

# 31. 下一个排列
### 方法1
```
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int n = nums.size();
        int k = n - 1;
        while (k > 0 && nums[k - 1] >= nums[k]) k--;
        if (k == 0) {
            reverse(nums.begin(), nums.end());
            return;
        }

        int j = k - 1;
        while (j + 1 < n && nums[j + 1] > nums[k - 1]) j++;
        swap(nums[k - 1], nums[j]);
        reverse(nums.begin() + k, nums.end());
    }
};
```

# 460. LFU
```
class LFUCache {
public:
    struct Node {
        Node *left, *right;
        int key, val;
        Node(int _key, int _val) :key(_key), val(_val)
        {
            left = right = nullptr;
        }
    };

    struct Block {
        Node *head, *tail;
        Block *left, *right;
        int cnt;
        Block(int _cnt)
        {
            cnt = _cnt;
            left = right = nullptr;
            head = new Node(-1, -1);
            tail = new Node(-1, -1);
            head->right = tail;
            tail->left = head;
        }

        void insert(Node* p)
        {   
            p->right = head->right;
            head->right->left = p;
            p->left = head;
            head->right = p;
        }

        void remove(Node* p)
        {
            p->left->right = p->right;
            p->right->left = p->left;
            // p->left->right = p->right;
        }

        bool empty()
        {
            return head->right == tail;
        }
    };

    Block *head, *tail;
    int n;
    unordered_map<int, Block*> hash_block;
    unordered_map<int, Node*> hash_node;

    LFUCache(int capacity) {
        n = capacity;
        head = new Block(0), tail = new Block(INT_MAX);
        head->right = tail;
        tail->left = head;
    }

    void insert(Block* p)
    {
        Block* cur = new Block(p->cnt + 1);
        cur->right = p->right;
        p->right->left = cur;
        cur->left = p;
        p->right = cur;
    }

    void remove(Block* p)
    {
        p->left->right = p->right;
        p->right->left = p->left;
        delete p;
    }
    
    int get(int key) {
        if (hash_block.count(key) == 0) return -1;
        auto block = hash_block[key];
        auto node = hash_node[key];
        block->remove(node);
        if (block->right->cnt != block->cnt + 1) {
            insert(block);
        }
        block->right->insert(node);
        hash_block[key] = block->right;
        if (block->empty()) remove(block);

        return node->val;
    }
    
    void put(int key, int value) {
        if (n == 0) return;
        if (hash_node.count(key)) {
            hash_node[key]->val = value;
            get(key);
        } else {
            if (hash_node.size() == n) {
                auto p = head->right->tail->left;
                head->right->remove(p);
                if (head->right->empty()) remove(head->right);
                hash_block.erase(p->key);
                hash_node.erase(p->key);
                delete p;
            }

            auto p = new Node(key, value);
            if (head->right->cnt > 1) {
                insert(head);
            }
            head->right->insert(p);
            hash_node[key] = p;
            hash_block[key] = head->right;
        }
    }
};
 ```
 
 # 1143. 最长公共子序列
 ```
class Solution {
public:
    int longestCommonSubsequence(string text1, string text2) {
        int n = text1.size(), m = text2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1, 0));

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <=m; j++) {
                if (text1[i - 1] == text2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[n][m];
    }
};
```
# 22. 括号生成
```
class Solution {
public:
    vector<string> res;
    int m;
    vector<string> generateParenthesis(int n) {
        m = n;
        dfs("", 0, 0);

        return res;
    }

    void dfs(string s, int l, int r) {
        if (r > l || l > m || r > m) return;

        if (l == r && l == m) {
            res.push_back(s);
            return;
        }

        dfs(s + '(', l + 1, r);
        dfs(s + ')', l, r + 1);
    }
};
```
# 151. 翻转字符串里的单词
```
class Solution {
public:
    string reverseWords(string s) {
        int i = 0;
        while (i < s.size() && s[i] == ' ') i++;
        s.erase(s.begin(), s.begin() + i);
        while (s.size() && s.back()  == ' ') s.pop_back();
        reverse(s.begin(), s.end());
        int k = 0, j = 0;
        for (i = 0; i < s.size(); ) 
        {
            if (s[i] != ' ') {
                s[k++] = s[i++];
            } 
            else 
            {
                s[k++] = s[i++];
                int j = i;
                while (j < s.size() && s[j] == ' ') j++;
                i = j;
            }
        }
        s.erase(s.begin() + k, s.end());

        for (int i = 0; i < s.size(); ) {
            int j = i;
            while (j < s.size() && s[j] != ' ') j++;
            reverse(s.begin() + i, s.begin() + j);
            i = j + 1;
        }
        return s;
    }
};
```
# 8. 字符串转换整数 (atoi)
### 方法1
```
class Solution {
public:
    int myAtoi(string s) {
        string res;
        int f = 1, i = 0;

       while (i < s.size() && s[i] == ' ') i++;
       if (s[i] == '+') {
           f = 1;
           i++;
       } else if (s[i] == '-') {
           f = -1;
           i++;
       }
       while (i < s.size()) {
        //    if (s[i] >= '0' & s[i] <= '9') {
           if (isdigit(s[i])) {
               res += s[i];
               i++;
               if (f == 1 && stol(res) >=INT_MAX) {
                   return INT_MAX;
               } else if (f == -1 && stol(res) * (-1) <= INT_MIN) {
                   return INT_MIN;
               }
           } else break;
       }
        if (res.size() == 0) return 0;
        return stoi(res) * f;
    }
};
```

### 方法2
```
class Solution {
public:
    int myAtoi(string s) {
        int res = 0;
        int f = 1, i = 0;

       while (i < s.size() && s[i] == ' ') i++;
       if (s[i] == '+') {
           f = 1;
           i++;
       } else if (s[i] == '-') {
           f = -1;
           i++;
       }
       
       while (i < s.size()) {
           if (s[i] >= '0' & s[i] <= '9') {
               int x = s[i] - '0';
               i++;
               if (f == 1 && res > (INT_MAX - x) / 10) {
                   return INT_MAX;
               } else if (f == -1 && (-1 * res ) < (INT_MIN + x) / 10 ) {
                   return INT_MIN;
               } else if (-1 * res * 10 - x == INT_MIN) {
                   return INT_MIN;
               }
               res = res * 10 + x;

           } else break;
       }
        
        return res * f;
    }
};
```

# 239. 滑动窗口最大值
```
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        vector<int> res;
        deque<int> dq;
        for (int i = 0; i < nums.size(); i++) {
            if (dq.size() && i - dq.front() >= k) dq.pop_front();
            while (dq.size() && nums[dq.back()] < nums[i]) dq.pop_back();
            dq.push_back(i);
            if (i + 1 >= k) {
                // cout << "i: " << i << ", k: " << k << ", front: " << dq.front() << endl;
                res.push_back(nums[dq[0]]);
            }
        }

        return res;
    }
};
```
# 41. 缺失的第一个正数
```class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i])
                swap(nums[i], nums[nums[i] - 1]);
        }

        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }

        return n + 1;
    }
};
```

# 剑指 Offer 22. 链表中倒数第k个节点
```
/**
 * Definition for singly-linked list.
 * struct ListNode {
 *     int val;
 *     ListNode *next;
 *     ListNode(int x) : val(x), next(NULL) {}
 * };
 */
class Solution {
public:
    ListNode* getKthFromEnd(ListNode* head, int k) {
        auto fast = head, slow = head;
        for (int i = 1; i <= k; i++) fast = fast->next;
        while (fast) {
            slow = slow->next;
            fast = fast->next;
        }

        return slow;
    }
};
```
# 93. 复原IP地址
```
class Solution {
public:
    vector<string> ans;
    vector<string> path;

    vector<string> restoreIpAddresses(string s) {
        dfs(s, 0);

        return ans;
    }

     bool isValid(string& s) {
        int len = s.size();
        long long ans = 0, numZero = 0, preZero = 0;
        for (int i = 0; i < len; i++) {
            if (s[i] < '0' || s[i] > '9') return false;
            else {
                ans = ans * 10 + s[i] - '0';
                if (s[i] == '0' && i == 0) numZero++;
            }
        }

        return (ans == 0 && s.size() == 1) || (numZero == 0 && ans > 0 && ans <= 255);
    }

    void dfs(string& s, int u)
    {
        if (u == s.size() && path.size() == 4) {
            string temp;
            for (int i = 0; i < 3; i++) 
                temp += (path[i] + '.');
            temp += path[3];
            ans.push_back(temp);
            return;
        }

        for (int i = u; i < s.size(); i++) {
            string temp = s.substr(u, i - u + 1);
            if (isValid(temp)) {
                path.push_back(temp);
                dfs(s, i + 1);
                path.pop_back();
            }
        }
    }
};
```
# 105. 从前序与中序遍历序列构造二叉树
```
class Solution {
public:
    unordered_map<int, int> hash;
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for (int i = 0; i < inorder.size(); i++) {
            hash[inorder[i]] = i;
        }

        return build(preorder, 0, preorder.size() - 1, inorder, 0, inorder.size() - 1);
    }

    TreeNode* build(vector<int>& pre, int pl, int pr, vector<int>& in, int il, int ir)
    {
        if (pl > pr) return nullptr;
        auto root = new TreeNode(pre[pl]);
        int pos = hash[pre[pl]];
        root->left = build(pre, pl + 1, pl + 1 + pos - 1 - il, in, il, pos - 1);
        root->right = build(pre, pl + 1 + pos - 1 - il + 1, pr, in, pos + 1, ir);

        return root;
    }
};
```