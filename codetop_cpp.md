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
### 方法2：更简洁的写法
```
class Solution {
public:
    string addStrings(string num1, string num2)
    {
        int carry = 0;
        int i = num1.size() - 1, j = num2.size() - 1;
        string res;
        while (i >= 0 || j >= 0 || carry)
        {
            int x = i >= 0 ? num1[i] - '0' : 0;
            int y = j >= 0 ? num2[j] - '0' : 0;
            int temp = x + y + carry;
            res += '0' + temp % 10;
            carry = temp / 10;
            i--, j--;
        }
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


# 165. 比较版本号
```
class Solution {
public:

    vector<int> getVans(string version1)
    {
        vector<int> v1, v2;
        int res = 0;
        for (int i = 0; i < version1.size(); i++) {
            if (version1[i] != '.') {
                // cout << "res = " << res << endl;
                res = res * 10 - '0' + version1[i]; //先减后加，防止溢出
            } else {
                v1.push_back(res);
                res = 0;
            }
        }
        v1.push_back(res);

        return v1;
    }

    int compareVersion(string version1, string version2) {
        vector<int> v1 = getVans(version1);
        vector<int> v2 = getVans(version2);
        int i = 0;
        while (i < v1.size() && i < v2.size()) {
            if (v1[i] > v2[i]) return 1;
            else if (v1[i] < v2[i]) return -1;
            else i++;
        }

        if (i == v1.size() && i == v2.size()) return 0;
        if (i == v1.size()) {
            while (i < v2.size() && v2[i] == 0) i++;
            if (v2.size() == i) return 0;
            return -1;
        }

        if (i == v2.size()) {
            while (i < v1.size() && v1[i] == 0) i++;
            if (v1.size() == i) return 0;
            return 1;
        }

        return 0;  
    }
};
```
# 76. 最小覆盖子串
```
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char, int> ht, hs;
        for (auto x : t) ht[x]++;
        int cnt = 0;
        string res;
        for (int i = 0, j = 0; i < s.size(); i++) {
            hs[s[i]]++;
            if (hs[s[i]] <= ht[s[i]]) cnt++;
            while (hs[s[j]] > ht[s[j]]) hs[s[j++]]--;
            if (cnt == t.size()) {
                if (res.size() == 0 || i - j + 1 < res.size()) {
                    res = s.substr(j, i - j + 1);
                }
            }
        }

        return res;
    }
};
```
# 43. 字符串相乘
```
class Solution {
public:
    string multiply(string num1, string num2) {
        if (num1 == "0" || num2 == "0") return "0";
        if (num1 == "1") return num2; 
        if (num2 == "1") return num1;
        vector<int> a, b;
        for (int i = num1.size() - 1; i >= 0; i--) a.push_back(num1[i] - '0'); 
        for (int i = num2.size() - 1; i >= 0; i--) b.push_back(num2[i] - '0');
        vector<int> c(num1.size() + num2.size());

        for (int i = 0; i < num1.size(); i++)
            for (int j = 0; j < num2.size(); j++) {
                c[i + j] += a[i] * b[j];
            } 
        int t = 0;
        for (int i = 0; i < num2.size() + num1.size(); i++) {
            c[i] += t;
            t = c[i] / 10;
            c[i] = c[i] % 10;
        }
        int k = num1.size() +  num2.size() - 1;

        while (k >= 0 && c[k] == 0) k--;
        string res;
        while (k >= 0) res += c[k--] + '0';
        return res;
    }
};
```
# 155. 最小栈
```
class MinStack {
public:
    stack<int> st, stMin;

    MinStack() {

    }
    
    void push(int val) {
        if (st.empty()) {
            st.push(val);
            stMin.push(val);
        } else {
            if (val > stMin.top()) {
                stMin.push(stMin.top());
                st.push(val);
            } else {
                stMin.push(val);
                st.push(val);
            }
        }
    }
    
    void pop() {
        if (st.size()) {
            st.pop();
            stMin.pop();
        }
    }
    
    int top() {
        return st.top();
    }
    
    int getMin() {
        return stMin.top();
    }
};

/**
 * Your MinStack object will be instantiated and called as such:
 * MinStack* obj = new MinStack();
 * obj->push(val);
 * obj->pop();
 * int param_3 = obj->top();
 * int param_4 = obj->getMin();
 */
 ```
# 110. 平衡二叉树
 ```
class Solution {
public:
    bool ans;
    bool isBalanced(TreeNode* root) {
        if (!root) return true;
        ans = true;
        dfs(root);

        return ans;
    }

    int dfs(TreeNode* root) {
        if (!root) return 0;
        int l = dfs(root->left);
        int r = dfs(root->right);
        if (abs(l - r) > 1) ans = false;

        return max(l, r) + 1;
    }
};
```

### 方法2
```
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode() : val(0), left(nullptr), right(nullptr) {}
 *     TreeNode(int x) : val(x), left(nullptr), right(nullptr) {}
 *     TreeNode(int x, TreeNode *left, TreeNode *right) : val(x), left(left), right(right) {}
 * };
 */
class Solution {
public:
    bool isBalanced(TreeNode* root) {
        if (!root) return true;
        if (abs(maxDepth(root->left) - maxDepth(root->right)) > 1) {
            return false;
        }

        if (isBalanced(root->left) == false) {
            return false;
        }

        if (isBalanced(root->right) == false) {
            return false;
        }


        return true;
    }
    int maxDepth(TreeNode* root) {
        if (!root) return 0;

        return max(maxDepth(root->left), maxDepth(root->right)) + 1;
    }
};
```
# 104. 二叉树的最大深度
```
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if (!root) return 0;

        int l = maxDepth(root->left);

        int r = maxDepth(root->right);

        return max(l, r) + 1;
    }
};
```
# 543. 二叉树的直径
### 方法1
```
class Solution {
public:
    int ans;
    int diameterOfBinaryTree(TreeNode* root) {
        if (!root) return 0;
        ans = 0;
        dfs(root);

        return ans;
    }
    void dfs(TreeNode* root)
    {
        if (!root) return;
        int l = maxDepth(root->left);
        int r = maxDepth(root->right);
        ans = max(ans, l + r);
        dfs(root->left);
        dfs(root->right);
    }
    int maxDepth(TreeNode* root) {
        if (!root) return 0;

        int l = maxDepth(root->left);

        int r = maxDepth(root->right);

        return max(l, r) + 1;
    }
};
```
### 方法2
```
class Solution {
public:
    int ans = 0;
    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return ans;
    }

    int dfs(TreeNode* root) {
        if (!root) return 0;
        int l = dfs(root->left);
        int r = dfs(root->right);
        ans = max(ans, l + r);;
        
        return 1 + max(l, r);
    }

};
```
# 78. 子集
### 方法1
```
// 2022.11.04 周五
class Solution {
public:
    vector<vector<int>> subsets(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> res;

        for (int i = 0; i < (1 << n); i++) {
            vector<int> t;
            for (int j = 0; j < n; j++) {
                if ((i >> j) & 1) {
                    t.push_back(nums[j]);
                }
            }
            res.push_back(t);
        }

        return res;
    }
};
```
### 方法2
```
// 2022.11.04 周五
class Solution {
public:
    int n;
    vector<vector<int>> res;
    vector<int> path;
    vector<vector<int>> subsets(vector<int>& nums) {
        n = nums.size();

        dfs(nums, 0);
        return res;
    }

    void dfs(vector<int>& nums, int u)
    {
        if (u == n) {
            res.push_back(path);
            // cout << "path: " << u << endl;
            return;
        }
        path.push_back(nums[u]);

            // cout << "i = " << i << endl;
        dfs(nums, u + 1);
        path.pop_back();
            // cout << "ii = " << i << endl;
        dfs(nums, u + 1);
        
    }
};
```
# 129. 求根节点到叶节点数字之和
```
class Solution {
public:
    vector<int> num;
    int sumNumbers(TreeNode* root) {
        dfs(root, 0);
        int res = 0;
        for (auto x : num) {
            // cout << "x: " << x << endl;
            res += x;
        }

        return res;
    }

    void dfs(TreeNode* root, int t) {
        if (root->left == nullptr && root->right == nullptr) {
            t = t * 10 + root->val;
            num.push_back(t);
            return;
        }

        t = t * 10 + root->val;
        if(root->left) dfs(root->left, t);
        if(root->right) dfs(root->right, t);
    }
};
```
# 32. 最长有效括号
## 方法1 栈
```
class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> st;
        int res = 0, l = -1, i = 0;
        for (auto x : s) {
            if (x == '(') {
                st.push(i);
            } else {
                if (st.empty()) {
                    l = i;
                } else {
                    st.pop();
                    if (st.empty()) {
                        res = max(res, i - l);
                    } else {
                        res = max(res, i - st.top());
                    }
                }
            }
            i++;
        }

        return res;
    }
};
```

## 方法2 两次遍历
```
class Solution {
public:
    int longestValidParentheses(string s) {
        int l = 0, r = 0, res = 0, len = s.size();
        for (int i = 0; i < len; i++) {
            if (s[i] == '(') l++;
            else r++;
            if (l == r) res = max(res, r + r);
            if (r > l) l = r = 0;
        }

        l = r = 0;
        for (int i = len - 1; i >= 0; i--) {
            if (s[i] == ')') r++;
            else l++;
            if (l == r) res = max(res, l + l);
            if (l > r) l = r = 0;
        }

        return res;
    }
};
```
## 方法3 dp
```
class Solution {
public:
    int longestValidParentheses(string s) {
        int n = s.size();
        vector<int> f(n);
        int res = 0, pre = 0;

        for (int i = 1; i < n; i++) {
            if (s[i] == ')') {
                pre = i - f[i - 1] - 1;
                if (pre >= 0 && s[pre] == '(') {
                    f[i] = f[i - 1] + 2;
                    if (pre) f[i] += f[pre - 1];
                }
            }
            res = max(res, f[i]);
        }

        return res;
    }
};
```

# 101. 对称二叉树
```
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if (!root) return true;
        return isMatch(root->left, root->right);
    }

    bool isMatch(TreeNode* p1, TreeNode* p2) {
        if (!p1 && !p2) return true;
        if (!p1 || !p2) return false;
        if (p1->val != p2->val) return false;
        return isMatch(p1->left, p2->right) && isMatch(p1->right, p2->left);
    }
};
```
# 64. 最小路径和
## dp
```
class Solution {
public:
    int minPathSum(vector<vector<int>>& grid) {
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};

        int m = grid.size(), n = grid[0].size();
        vector<vector<int>> dp(m, vector<int>(n));
        dp[0][0] = grid[0][0];
        for (int i = 1; i < n; i++) dp[0][i] = dp[0][i - 1] + grid[0][i];
        for (int i = 1; i < m; i++) dp[i][0] = dp[i - 1][0] + grid[i][0];
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1]) + grid[i][j];
                // cout << i << " " << j << " " << dp[i][j] << endl;
            }
        }

        return dp[m - 1][n - 1];
    }
};
```
# 322. 零钱兑换
### 方法1 完全背包，二维DP
```
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<vector<int>> dp(n + 1, vector<int>(amount + 1, 0x3f3f3f3f));

        for (int i = 0; i <= n; i++) {
            dp[i][0] = 0;
        }

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= amount; j++) {
                dp[i][j] = min(dp[i][j], dp[i - 1][j]);
                if (j >= coins[i - 1]) {
                    dp[i][j] = min(dp[i][j], dp[i][j - coins[i - 1]] + 1);
                }
            }
        }

        return dp[n][amount] == 0x3f3f3f3f ? -1 : dp[n][amount];
    }
};
```
### 方法2 优化空间
```
class Solution {
public:
    int coinChange(vector<int>& coins, int amount) {
        int n = coins.size();
        vector<int> dp(amount + 1, 0x3f3f3f3f);
        dp[0] = 0; // 这句优化时不能忘记


        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= amount; j++) {
                // dp[i][j] = min(dp[i][j], dp[i - 1][j]);
                dp[j] = min(dp[j], dp[j]);
                if (j >= coins[i - 1]) {
                    // dp[i][j] = min(dp[i][j], dp[i][j - coins[i - 1]] + 1);
                    dp[j] = min(dp[j], dp[j - coins[i - 1]] + 1);
                }
            }
        }

        return dp[amount] == 0x3f3f3f3f ? -1 : dp[amount];
    }
};
```
# 98. 验证二叉搜索树
### 方法1
```
class Solution {
public:
    vector<int> num;
    bool isValidBST(TreeNode* root) {
        dfs(root);

        for (int i = 1; i < num.size(); i++) {
            if (num[i - 1] >= num[i]) return false;
        }

        return true;
    }

    void dfs(TreeNode* root)
    {
        if (!root) return;

        dfs(root->left);
        num.push_back(root->val);
        dfs(root->right);
    }
};
```
### 方法2
```
class Solution {
public:

    bool ans;
    bool isValidBST(TreeNode* root) {
        ans = true;
        TreeNode* pre = nullptr;
        dfs(root, pre);

        return ans;
    }

    void dfs(TreeNode* root, TreeNode*& pre)
    {
        if (!root) return;

        dfs(root->left, pre);
        if (pre && pre->val >= root->val) ans = false;
        pre = root;
        dfs(root->right, pre);
    }
};
```

### 方法3
```
class Solution {
public:
    TreeNode* pre = nullptr;
    bool isValidBST(TreeNode* root) {
        if (!root) return true;

        if (!isValidBST(root->left)) return false;
        if (pre && root->val <= pre->val) return false;
        pre = root;
        if (!isValidBST(root->right)) return false;

        return true;
    }
};
```
### 方法4
```
class Solution {
public:
    vector<int> num;
    bool isValidBST(TreeNode* root) {
        Tarverse(root);

        for (int i = 1; i < num.size(); i++) {
            if (num[i - 1] >= num[i]) return false;
        }

        return true;
    }

    void Tarverse(TreeNode* root)
    {
        stack<TreeNode*> st;

        while (st.size() || root) {
            if (root) {
                st.push(root);
                root = root->left;
            } else {
                auto node = st.top(); 
                st.pop();
                num.push_back(node->val);
                root = node->right;
            }
        }
    }
};
```
# 113. 路径总和 II
```
class Solution {
public:
    vector<vector<int>> res;
    vector<vector<int>> pathSum(TreeNode* root, int targetSum) {
        if (!root) return res;
        vector<int> p;
        dfs(root, p, 0, targetSum);

        return res;
    }

    void dfs(TreeNode* root, vector<int> path, int s, int t)
    {
        if (root->left == nullptr && root->right == nullptr)
        {
            path.push_back(root->val);
            s += root->val;
            if (s == t) {
                res.push_back(path);
            }

            return;
        }
        s += root->val;
        path.push_back(root->val);
        if (root->left) dfs(root->left, path, s, t);
        if (root->right) dfs(root->right, path, s, t);
    }
};
```

# 470. 用 Rand7() 实现 Rand10()
## 方法1
```
class Solution {
public:
    int rand10() {
        int first, second;
        while ((first = rand7()) > 6);
        while ((second = rand7()) > 5);

        return (first % 2 == 1) ? second : second + 5;
    }
};
```

## 方法2
```
class Solution {
public:
    int rand10() {
        int t = (rand7() - 1) * 7 + rand7();
        if (t > 40) return rand10();

        return (t - 1) % 10 + 1;
    }
};
```
# 39. 组合总和
### 方法1
```
class Solution {
public:
    vector<vector<int>> res;
    vector<int> path;

    vector<vector<int>> combinationSum(vector<int>& nums, int t) {
        sort(nums.begin(), nums.end());
        dfs(nums, t, 0, 0);

        return res;
    }

    void dfs(vector<int>& nums, int t, int sum, int u)
    {
        if (sum > t) return;
        if (sum == t) {
            res.push_back(path);
            return;
        }
        
        for (int i = u; i < nums.size(); i++) {
            path.push_back(nums[i]);
            dfs(nums, t, sum + nums[i], i);
            path.pop_back();
        }

       
    }
};
```

### 方法2
```
class Solution {
public:
    vector<vector<int>> res;

    vector<vector<int>> combinationSum(vector<int>& nums, int t) {
        sort(nums.begin(), nums.end());
        vector<int> path;
        dfs(nums, t, 0, 0, path);

        return res;
    }

    void dfs(vector<int>& nums, int t, int sum, int u, vector<int> path)
    {
        if (sum > t) return;
        if (sum == t) {
            res.push_back(path);
            return;
        }
        if (u == nums.size()) return;
        
        while (sum <= t) {
            dfs(nums, t, sum, u + 1, path);
            sum += nums[u];
            path.push_back(nums[u]);
        }
    }
};
```
# 48. 旋转图像
```
class Solution {
public:
    void rotate(vector<vector<int>>& mat) {
        int n = mat.size(), m = mat[0].size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < m; j++) {
                swap(mat[i][j], mat[j][i]);
            }
        }

        for (int i = 0; i < n; i++) {
            int l = 0, r = m - 1;
            while (l < r) {
                swap(mat[i][l], mat[i][r]);
                l++, r--;
            }
        }
    }
};
```

# 112. 路径总和
```
class Solution {
public:
    bool hasPathSum(TreeNode* root, int targetSum) {
        if (root == nullptr) return false;

        if (root->left == nullptr && root->right == nullptr)
            return root->val == targetSum;
        
        return hasPathSum(root->left, targetSum - root->val) ||
               hasPathSum(root->right, targetSum - root->val);
    }
};
```
# 169. 多数元素
```
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int cnt = 0, res = 0;
        for (auto x : nums) {
            if (cnt == 0) res = x, cnt++;
            else {
                if (x == res) cnt++;
                else cnt--;
            }
        }

        return res;
    }
};
```
# 221. 最大正方形
```
class Solution {
public:
    int maximalSquare(vector<vector<char>>& mat) {
        int n = mat.size(), m = mat[0].size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1));
        int res = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (mat[i - 1][j - 1] == '1') {
                    f[i][j] = min(f[i - 1][j - 1], min(f[i][j - 1], f[i - 1][j])) + 1;
                }
                res = max(res, f[i][j]);  
            }
        }

        return res * res;
    }
};
```

# 240. 搜索二维矩阵 II
## 方法1
```
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int n = matrix.size(), m = matrix[0].size();
        int i = 0, j = m - 1;
        while (i >= 0 && i < n && j >= 0 && j < m)
        {
            if (matrix[i][j] > target) j--;
            else if (matrix[i][j] < target) i++;
            else return true;
        }

        return false;
    }
};
```
## 方法2
```
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int n = matrix.size(), m = matrix[0].size();
        
        for (int i = 0; i < n; i++)
        {
            int id = lower_bound(matrix[i].begin(), matrix[i].end(), target) - matrix[i].begin();
            if (id < m && matrix[i][id] == target) return true;
        }

        return false;
    }
};
```

# 226. 翻转二叉树
```
class Solution {
public:
    TreeNode* invertTree(TreeNode* root) {
        if (!root) return root;
        auto l = invertTree(root->left);
        auto r = invertTree(root->right);

        root->left = r;
        root->right = l;

        return root;
    }
};
```

# 718. 最长重复子数组
```
class Solution {
public:
    int findLength(vector<int>& a, vector<int>& b) {
        int n = a.size(), m = b.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        int res = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (a[i - 1] == b[j - 1]) {
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1);
                } 

                res = max(res, dp[i][j]);
                // else {
                //     dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
                // }
            }
        }

        return res;
    }
};
```

# 234. 回文链表
### 方法1
```
class Solution {
public:
    bool isPalindrome(ListNode* h) {
        vector<int> v;
        while (h) {
            v.push_back(h->val);
            h = h->next;
        }

        for (int i = 0, j = v.size() - 1; i < j; i++, j--) {
            if (v[i] != v[j]) return false;
        }

        return true;
    }
};
```
### 方法2 O(1)空间
```
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        int n = 0;
        for (auto p = head; p; p = p->next) {
            n++;
        }
        if (n <= 1) return head;
        int half = n / 2;
        auto a = head;

        for (int i = 0; i < n - half; i++) a = a->next;
        auto b = a->next;
        while (b) {
            auto c = b->next;
            b->next = a;
            a = b;
            b = c;
        }
        
        auto p = head, q = a;
        bool flag = true;
        for (int i = 0; i < half; i++) {
            if (p->val != q->val) {
                flag = false;
                break;
            }
            p = p->next;
            q = q->next;
        }

        auto tail = a;
        b = a->next;
        for (int i = 0; i < half - 1; i++) {
            auto c = b->next;
            b->next = a;
            a = b;
            b = c;
        }

        tail->next = nullptr;
        return flag;
    }
};
```

### 方法3
```
class Solution {
public:
    pair<ListNode*, ListNode*> getMidNode(ListNode* head, int n)
    {
        auto p = head;
        for (int i = 1; i < n/2; i++) {
            p = p->next;
        }
        auto res = p->next;
        p->next = nullptr;
        return make_pair(p, res);
    }

    ListNode* reverseList(ListNode* head)
    {
        if (!head || !head->next) return head;
        ListNode *a = head, *b = head->next, *c = nullptr;
        while (b) {
            c = b->next;
            b->next = a;
            a = b;
            b = c;
        }
        head->next = nullptr;

        return a;
    }

    bool isPalindrome(ListNode* head) {
        auto p = head;
        int n = 0;
        while (p) {
            n++;
            p = p->next;
        }
        p = head;
        auto midPair = getMidNode(head, n);
        auto midNode = midPair.second;
        auto pre= midPair.first;
        // cout << "pre->val: " << pre->val << ", mid->val: " << midNode->val << endl;
        // cout << "pre->next: " << pre->next <<  endl;
        auto rMidNode = reverseList(midNode);

        // for (ListNode* pt = rMidNode; pt; pt = pt->next) {
        //     cout << "pt->val: " << pt->val << endl;
        // }
        // cout << "rMidNode->val: " << rMidNode->val << endl;
        bool ans = true;
        while (p && rMidNode) {
            // cout << "----p->val: " << p->val << ", rmid->val: " << rMidNode->val << endl;
            if (p->val == rMidNode->val) {
                p = p->next, rMidNode = rMidNode->next;
            } else {
                ans = false;
                break;
            }
        }
        midNode = reverseList(rMidNode);
        pre->next = midNode;

        return ans;
    }
};
```


# 394. 字符串解码
## 方法1：栈实现
```
class Solution {
public:
    string decodeString(string s) {
        stack<char> st;
        for (auto x : s) {
            if (st.empty()) {
                st.push(x);
            } else {
                if (x != ']') {
                    st.push(x);
                } else {
                    stack<char> temp;
                    while (st.size() && st.top() != '[') {
                        temp.push(st.top());
                        st.pop();
                    }
                    st.pop();
                    string cntString;
                    while (st.size()) {
                        char ch = st.top();
                        if (ch >= '0' && ch <= '9') {
                            cntString += ch;
                        } else {
                            break;
                        }
                        st.pop();
                    }
                    // cout << "cntString: " << cntString << endl;
                    reverse(cntString.begin(), cntString.end());
                    int cnt = stoi(cntString);
                    if (cnt) {
                        string res;
                        while (temp.size()) {
                            res += temp.top();
                            temp.pop();
                        }
                        for (int i = 0; i < cnt; i++) {
                            for (int j = 0; j < res.size(); j++) {
                                st.push(res[j]);
                            }
                        }
                    }
                }
            }
        }

        string ans;
        while (st.size()) {
            ans += st.top();
            st.pop();
        }
        reverse(ans.begin(), ans.end());

        return ans;
    }
};
```
### 方法2 dfs
```
class Solution {
public:
    string decodeString(string s) {
        int u = 0;
        return dfs(s, u);
    }

    string dfs(string& s, int& u)
    {
        string res;
        while (u < s.size() && s[u] != ']') {
            if (s[u] >= 'a' && s[u] <= 'z') res += s[u++];
            else if (s[u] >= '0' && s[u] <= '9') {
                int k = u;
                while (s[k] >= '0' && s[k] <= '9') k++;
                int cnt = stoi(s.substr(u, k - u));
                u = k + 1;
                string y = dfs(s, u);
                u++;
                while (cnt--) res += y;
            }
        }

        return res;
    }
};
```

# 34. 在排序数组中查找元素的第一个和最后一个位置
## 方法1
```
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int id1 = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
        if (id1 == nums.size()) return {-1, -1};
        int id2 = upper_bound(nums.begin(), nums.end(), target) - nums.begin();
        if (nums[id1] == target) {
            return {id1, id2 - 1};
        }

        return {-1, -1};
    }
};
```
## 方法2
```
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if (nums.size() == 0) return {-1, -1};
        int l = 0, r = nums.size() - 1;
        int a = -1, b = -1;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }

        if (nums[l] != target) return {-1, -1};
        a = l;
        l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] > target) r = mid;
            else l = mid + 1;
        }
        if (nums[r] == target) b = r;
        else if (r >= 1 && nums[r - 1] == target) b = r - 1;

        return {a, b};
    }
};
```
## 方法3
```
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if (nums.size() == 0) return {-1, -1};
        int l = 0, r = nums.size() - 1;
        int a = -1, b = -1;
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }

        if (nums[l] != target) return {-1, -1};
        a = l;
        l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = (l + r + 1) >> 1;
            if (nums[mid] > target) r = mid - 1;
            else l = mid;
        }
        if (nums[r] == target) b = r;
        // else if (r >= 1 && nums[r - 1] == target) b = r - 1;

        return {a, b};
    }
};
```
# 14. 最长公共前缀
```
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        for (int i = 0; i < strs[0].size(); i++) {
            for (int j = 1; j < strs.size(); j++) {
                if (strs[j][i] != strs[0][i]) {
                    return strs[0].substr(0, i);
                }
            }
        }

        return strs[0];
    }
};
```
# 162. 寻找峰值
[//]: # (支持粘贴图片啦🎉🎉🎉)
[//]: # (保存的笔记可以在 CodeTop 微信小程序中查看)
## 方法1 O(n)
```
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        if (nums.size() == 1) return 0;
        for (int i = 0; i < nums.size(); i++) {
            if (i == 0 && nums[i] > nums[i + 1]) return i;
            else if ((i == nums.size() - 1) && nums[i] > nums[i - 1]) return i;
            else if (i && i < nums.size() - 1 && nums[i] > nums[i - 1] && nums[i] > nums[i + 1])
                return i;
        }

        return 0;
    }
};
```
## 方法2 二分
```
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] > nums[mid + 1]) r = mid;
            else l = mid + 1;
        }

        return l;
    }
};
```


# 128. 最长连续序列
### 方法1 暴力
```
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        nums.erase(unique(nums.begin(), nums.end()), nums.end());
        if (nums.size() == 1) return 1;
        int res = 0;

        for (int i = 1; i < nums.size(); i++) {
            int j = i;
            while (j < nums.size() && nums[j] - nums[j - 1] == 1) {
                j++;
            }
            res = max(res, j - i + 1);
            i = j;
        }

        return res;
    }
};
```

### 方法2 并查集
```
class Solution {
public:
    unordered_map<int, int> p, cnt;

    int find(int x) {
        if (x != p[x]) p[x] = find(p[x]);

        return p[x];
    }

    void merge(int a, int b) {
        int x = find(a);
        int y = find(b);
        if (x != y) {
            p[x] = y;
            cnt[y] += cnt[x];
        }
    }
    int longestConsecutive(vector<int>& nums) {
        for (auto x : nums) {
            p[x] = x;
            cnt[x] = 1;
        }

        for (auto x : nums) {
            if (p.count(x - 1)) {
                merge(x, x - 1);
            }
        }

        int res = 0;
        // for (auto x : nums) {
        //     res = max(res, cnt[x]);
        // }

        for (auto [k, v] : cnt) {
            res = max(res, v);
        }

        return res;

    }
};
```
### 方法3 哈希表
```
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        int res = 0;
        unordered_set<int> us(nums.begin(), nums.end());
        for (auto x : nums) {
            int l = x - 1, r = x + 1;
            while (us.count(l)) {
                us.erase(l);
                l--;
            }

            while (us.count(r)) {
                us.erase(r);
                r++;
            }

            res = max(res, r - l - 1);
        }

        return res;
    }
};
```
# 227. 基本计算器 II
```
class Solution {
public:
    stack<int> num;
    stack<char> op;

    int calculate(string s) {
        int n = s.size();
        unordered_map<char, int> mp;
        mp['+'] = mp['-'] = 1;
        mp['*'] = mp['/'] = 2;
        for (int i = 0; i < n; i++) {
            if (s[i] == ' ') continue;
            else if (isdigit(s[i])) {
                int j = i;
                int x = 0;
                while (j < n && isdigit(s[j])) {
                    x = x * 10 - '0'  + s[j];
                    j++;
                }
                num.push(x);
                i = j - 1;
            } else {
                while (op.size() && mp[s[i]] <= mp[op.top()]) {
                    cal();
                }
                op.push(s[i]);
            }
        }

        while (op.size()) cal();

        return num.top();
    }

    void cal()
    {
        char ch = op.top();
        op.pop();
        int a = num.top(); num.pop();
        int b = num.top(); num.pop();
        int c = 0;
        switch (ch) {
            case '+':
                c = a + b;
                break;
            case '-':
                c = b - a;
                break;
            case '*':
                c = b * a;
                break;
            case '/':
                c = b / a;
                break;
            default:
                break;
        }
        // if (ch == '+') {
        //     c = a + b;
        // } else if (ch == '-') {
        //     c = b - a;
        // } else if (ch == '*') {
        //     c = a * b;
        // } else if (ch == '/') {
        //     c = b / a;
        // }
        num.push(c);
    }
};
```

# 695. 岛屿的最大面积
## dfs
```
class Solution {
public:
    int n, m;
    vector<vector<int>> g;
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};

    int dfs(int x, int y) {
        g[x][y] = 0;
        int cnt = 1;
        for (int i = 0; i < 4; i++) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && g[a][b] == 1) {
                cnt += dfs(a, b);
            }
        }

        return cnt;
    }

    int maxAreaOfIsland(vector<vector<int>>& grid) {
        g = grid;
        int res = 0;
        n = g.size(), m = g[0].size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (g[i][j] == 1) {
                    // cout << "i = " << i << ", j = " << j << endl;
                    res = max(res, dfs(i, j));
                }
            }
        }

        return res;
    }
};
```
## bfs
```
class Solution {
public:
    int n, m;
    vector<vector<int>> g;
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};

    int dfs(int x, int y) {
        g[x][y] = 0;
        int cnt = 1;
        for (int i = 0; i < 4; i++) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && g[a][b] == 1) {
                cnt += dfs(a, b);
            }
        }

        return cnt;
    }

    int bfs(int x, int y) {
        int cnt = 0;
        queue<pair<int, int>> q;
        q.push(make_pair(x, y));
        g[x][y] = 0;
        while (q.size()) {
            int len = q.size();
            // cout << "len = " << len << endl;
            for (int i = 0; i < len; i++) {
                auto t = q.front();
                q.pop();
                int cx = t.first, cy = t.second;
                cnt++;
                // cout << "cx = " << cx << ", cy = " << cy << ", cnt = " << cnt << endl;
                
                for (int j = 0; j < 4; j++) {
                    int a = cx + dx[j], b = cy + dy[j];
                    if (a >= 0 && a < n && b >= 0 && b < m && g[a][b] == 1) {
                        q.push(make_pair(a, b));
                        g[a][b] = 0;
                    }
                }
            }
        }

        return cnt;
    }

    int maxAreaOfIsland(vector<vector<int>>& grid) {
        g = grid;
        int res = 0;
        n = g.size(), m = g[0].size();
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (g[i][j] == 1) {
                    res = max(res, bfs(i, j));
                    // cout << "i = " << i << ", j = " << j << ", res = " << res << endl;
                }
            }
        }

        return res;
    }
};
```
# 83. 删除排序链表中的重复元素
###  方法1
```
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        auto p = head;
        while (p) {
            auto c = p;
            while (c->next && c->next->val == p->val) {
                c = c->next;
            }

            p->next = c->next;
            p = p->next;;
        }

        return head;
    }
};
```
###  方法2
```
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        auto p = head;
        while (p) {
            auto c = p;
            while (c && c->val == p->val) {
                c = c->next;
            }

            p->next = c;
            p = p->next;;
        }

        return head;
    }
};
```

# 62. 不同路径
```
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> dp(m, vector<int>(n));

        for (int i = 0; i < n; i++) dp[0][i] = 1;
        for (int i = 0; i < m; i++) dp[i][0] = 1;
        for (int i = 1; i < m; i++) {
            for (int j = 1; j < n; j++) {
                dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
            }
        }

        return dp[m - 1][n - 1];
    }
};
```

# 198. 打家劫舍
```
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0];

        vector<int> dp(n);
        dp[0] = nums[0];
        dp[1] = max(dp[0], nums[1]);

        for (int i = 2; i < n; i++) {
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
        }

        return dp[n - 1];       
    }
};
```

```
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> dp(n, vector<int>(2));
        dp[0][0] = 0;
        dp[0][1] = nums[0];

        for (int i = 1; i < n; i++) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1]);
            dp[i][1] = dp[i - 1][0] + nums[i];
        }

        return dp[n - 1][0] > dp[n - 1][1] ? dp[n - 1][0] : dp[n - 1][1];
    }
};
```
# 122. 买卖股票的最佳时机 II
### 方法1
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res = 0;
        for (int i = 0, l = INT_MAX; i < prices.size(); i++) {
            if (prices[i] > l) {
                res += prices[i] - l;
                l = prices[i];
            } else {
                l = min(l, prices[i]);
            }
            
        }   

        return res;
    }
};
```
### 方法2
```
class Solution {
public:
    int maxProfit(vector<int>& p) {
        int l = 0, res = 0;
        for (int i = 1; i < p.size(); i++) {
            if (p[i] > p[i - 1]) res += p[i] - p[i - 1];
        }

        return res;
    }
};
```
# 24. 两两交换链表中的节点
```
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        if (!head || !head->next) return head;
        auto dummy = new ListNode();
        auto p = dummy;
        p->next = head;
        auto s = head, f = head->next;
        while (s && f) {
            auto t = f->next;
            f->next = s;
            s->next = t;
            p->next = f;
            p = s;
            s = f = nullptr;
            if (p && p->next) s = p->next;
            if (s && s->next) f = s->next;
        }

        return dummy->next;
    }
};
```
# 283. 移动零
```
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        int k = 0;
        for (int i = 0; i < nums.size(); i++) 
            if (nums[i]) nums[k++] = nums[i];

        for (; k < nums.size(); k++) nums[k] = 0;
    }
};
```
# 152. 乘积最大子数组
### 方法1
```
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int res = INT_MIN;
        int s = 1;
        for (int i = 0; i < nums.size(); i++) {
            s = s * nums[i];
            res = max(res, s);
            if (nums[i] == 0) s = 1;
        }
        s = 1;
        for (int i = nums.size() - 1; i >= 0; i--) {
            s = s * nums[i];
            res = max(res, s);
            if (nums[i] == 0) s = 1;
        }

        return res;
    }
};
```
### 方法2
```
class Solution {
public:
    int maxProduct(vector<int>& nums) {
        int f = nums[0], g = nums[0], res = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            int fa = nums[i] * f, ga = nums[i] * g;
            f = max(fa, max(ga, nums[i]));
            g = min(fa, min(ga, nums[i]));
            res = max(f, res);
        }

        return res;
    }
};
```
# 662. 二叉树最大宽度
```
// 2022.11.14 周一
class Solution {
public:
    struct Node {
        TreeNode* m_node;
        long long m_no;
        Node(TreeNode* _node, int _no):m_node(_node), m_no(_no)
        {

        }
    };

    int widthOfBinaryTree(TreeNode* root) {
        if (!root) {
            return 0;
        }
        queue<Node> q;
        q.push(Node(root, 1));
        long long res = 0;
        while (q.size()) {
            int len = q.size();
            long long l = LONG_MAX, r = LONG_MIN;
            for (int i = 0; i < len; i++) {
                auto t = q.front(); q.pop();
                l = min(l, t.m_no);
                r = max(r, t.m_no);
                res = max(res, r - l + 1);
                t.m_no -= l;
                if (t.m_node->left) {
                    q.push(Node(t.m_node->left, t.m_no * 2));
                }

                if (t.m_node->right) {
                    q.push(Node(t.m_node->right, t.m_no * 2 + 1));
                }
            }
        }

        return res;
    }
};
```

# 153. 寻找旋转排序数组中的最小值
```
class Solution {
public:
    int findMin(vector<int>& nums) {
        int l = 0, r = nums.size() - 1;
        if (nums[r] > nums[l]) return nums[l];
        while (l < r) {
            int mid = (l + r) >> 1;
            if (nums[mid] >= nums[0]) {
                l = mid + 1;
            } else {
                r = mid;
            }
        }

        return nums[l];
    }
};
```

# 138. 复制带随机指针的链表
```
class Solution {
public:
    Node* copyRandomList(Node* head) {
        auto dummy = new Node(0);
        dummy->next = head;
        auto ptr = dummy;
        auto p = head;
        while (p) {
            auto t = new Node(p->val);
            auto temp = p->next;
            p->next = t;
            t->next = temp;
            p = temp;
        }

        p = head;
        // cout << "p->val: " << p->val << endl;
        // cout << "p->random: " << p->random << endl;
        while (p) {
            if (p->random) // 这个判断不能少
                p->next->random = p->random->next;
            p = p->next->next;
        }

        p = head;
        while (p) {
            ptr->next = p->next;
            p->next = p->next->next;
            ptr = ptr->next;
            p = p->next;
        }

        return dummy->next;
    }
};
```
### 方法2 哈希表
```
class Solution {
public:
    Node* copyRandomList(Node* head) {
        unordered_map<Node*, Node*> h;

        auto p = head;
        while (p) {
            auto t = new Node(p->val);
            h[p] = t;
            p = p->next;
        }

        p = head;

        while (p) {
            h[p]->next = h[p->next];
            h[p]->random = h[p->random];
            p = p->next;
        }

        return h[head];
    }
};
```

# 179. 最大数
```
class Solution {
public:
    string largestNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end(), [](int x, int y) {
            string a = to_string(x);
            string b = to_string(y);

            return a + b > b + a;
        });

        string res;
        for (auto x : nums) {
            res += to_string(x);
        }
        int i = 0;
        while (i < res.size() - 1 && res[i] == '0') i++;
        res = res.substr(i);

        return res;
    }
};
```

# 209. 长度最小的子数组
### 方法1 暴力超时
```
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int res = INT_MAX;
        for (int i = 0; i < nums.size(); i++) {
            int j = i, s = 0;
            for (; j < nums.size(); j++) {
                s += nums[j];
                if (s >= target) {
                    res = min(res, j - i + 1);
                    break;
                } 
            }
        }

        return res == INT_MAX ? 0 : res;
    }
};
```
### 方法2 双指针
```
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int res = INT_MAX, s = 0;
        for (int i = 0, j = 0; i < nums.size(); i++) {
            s += nums[i];
            while (s >= target) {
                res = min(res, i - j + 1);
                s -= nums[j++];
            }
        }

        return res == INT_MAX ? 0 : res;
    }
};
```
# 39. 单词拆分
```
class Solution {
public:
    bool wordBreak(string s, vector<string>& wordDict) {
        int n = s.size();
        vector<bool> dp(n + 1);
        dp[0] = true;

        for (int i = 1; i <= n; i++) {
            for (auto &x : wordDict) {
                int len = x.size();
                if (i >= len && s.substr(i - len, len) == x) {
                    dp[i] = dp[i] || dp[i - len];
                }
            }
        }

        return dp[n];
    }
};
```

# 136. 只出现一次的数字
```
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        int res = 0;
        for (auto& x: nums) {
            res ^= x;
        }

        return res;
    }
};
```

# 560. 和为K的子数组
### 前缀和 + 哈希
```
class Solution {
public:
    int subarraySum(vector<int>& nums, int k) {
        unordered_map<int, int> h;
        h[0] = 1;
        int res = 0, s = 0;
        for (auto &x : nums) {
            s += x;
            res += h[s - k];
            h[s]++;
        }

        return res;
    }
};
```
# 498. 对角线遍历
### 注意边界条件
```
// 2022.11.17 周四
class Solution {
public:
    vector<int> findDiagonalOrder(vector<vector<int>>& a) {
        vector<int> res;
        int n = a.size(), m = a[0].size();
        // cout << "n: " << n << ", m: " << m << endl;
        int i = 0, j = 0, f = 1, k = 1;
        while (k < m + n) {
            // cout << "k: " << k << endl;
            if (f) {
                while (i >= 0 && j >= 0 && i < n && j < m) {
                    // cout << f << " " << i << " " << j << endl;
                    res.push_back(a[i][j]);
                    i--;j++;
                }
                f = 0;
                if (j == m) {
                    i += 2;
                    j--;
                }
                else {
                    i++;
                }
            } else {
                while (i >= 0 && j >= 0 && i < n && j < m) {
                    // cout << f << " " << i << " " << j << endl;
                    res.push_back(a[i][j]);
                    i++;j--;
                }
                f = 1;
                if (i == n) {
                    i --;
                    j += 2;
                } else j++;
            }
            k++;
        }

        return res;
    }
};
```
# 468. 验证IP地址
### 注意判断IP是否有效的技巧，加一个前导0的计数
```
class Solution {
public:
    string validIPAddress(string queryIP) {
        vector<string> num;
        int j = 0;
        for (int i = 0; i < queryIP.size(); i++) {
            if (queryIP[i] == '.' || queryIP[i] == ':') {
                // if (i && (queryIP[i - 1] == '.' || queryIP[i - 1] == ':')) return "Neither";
                num.push_back(queryIP.substr(j, i - j));
                j = i + 1;
            }
        }
        num.push_back(queryIP.substr(j));
        // cout << num.size() << endl;
        string res = "Neither";
        if (num.size() == 4 && isValidIPV4(num)) {
            res = "IPv4";
        } else if (num.size() == 8 && isValidIPV6(num)) {
            res = "IPv6";
        }

        return res;
    }

    bool isValidIPV4(vector<string>& num)
    {
        
        for (auto &x : num) {
            if (x.size() > 3 || x.size() == 0) return false;
            long long t = 0, zeroNo = 0;
            for (int i = 0; i < x.size(); i++) {
                if (x[i] < '0' || x[i] > '9') return false;
                int a = x[i] - '0';
                if (a == 0 && i == 0) zeroNo++;
                t = t * 10 + a;
            }

            if ((t == 0 && x.size() == 1) || (zeroNo == 0) && (t > 0 && t <= 255)) continue;
            else return false;
        }

        return true;
    }


    bool isValidIPV6(vector<string>& num)
    {
        for (auto& x : num) {
            int len = x.size();
            if (len > 4 || len == 0) return false;
            for (int i = 0; i < len; i++) {
                if ((x[i] >= '0' && x[i] <= '9') || 
                    (x[i] >= 'a' && x[i] <= 'f') || 
                    (x[i] >= 'A' && x[i] <= 'F') ) {
                        continue;
                    }
                else {
                    return false;
                }
            }
        }

        return true;
    }
};
```

# 剑指 Offer 09. 用两个栈实现队列
```
class CQueue {
public:
    stack<int> st1, st2;

    CQueue() {

    }
    
    void appendTail(int value) {
        st1.push(value);
    }
    
    int deleteHead() {
        if (st1.empty()) return -1;
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
};
```

# 912. 排序数组
### 堆排序的实现
```
class Solution {
public:
    int n;
    vector<int> h;

    vector<int> sortArray(vector<int>& nums) {
        h = nums;
        n = nums.size();
        vector<int> ans;

		// 这里从 n / 2开始是因为最后一层不需要建树
        for (int i = n / 2; i >= 0; i--) {
            down(i);
            // cout << "i: " << i << endl;
            // for (auto x : h) cout << x << " ";
            // cout << endl;
        }

        // cout << h[0] << endl;
        for (int i = 0; i < nums.size(); i++) {
            ans.push_back(h[0]);
            h[0] = h[n - 1];
            n--;
            down(0);
        }

        return ans;
    }

    void down(int u)
    {
        int t = u;
        // cout << "111, u = " << u << ", t = " << t << endl;
        if (2 * u + 1 < n && h[t] > h[2 * u + 1]) t = 2 * u + 1;
        if (2 * u + 2 < n && h[t] > h[2 * u + 2]) t = 2 * u + 2;
        // cout << "222, u = " << u << ", t = " << t << endl;
        if (u != t)
        {
            // cout << "------" << endl;
            swap(h[u], h[t]);
            down(t);
        }
    }
};
```

# 297. 二叉树的序列化与反序列化
### BFS
```
class Codec {
public:

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        if (!root) return "";

        string res;
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front(); q.pop();
                if (!node) {
                    res += "#,";
                    continue;
                }

                res += to_string(node->val) + ",";
                q.push(node->left);
                q.push(node->right);
            }
        }

        // cout << res << endl;

        return res;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        vector<string> v = splitString(data);

        // for (auto x : v) {
        //     cout << x << " ";
        // }
        // cout << endl;

        if (v.empty()) return NULL;
        queue<TreeNode*> q;
        auto root = new TreeNode(stoi(v[0]));
        q.push(root);

        for (int i = 1; i < v.size(); ) {
            auto node = q.front();
            if (v[i] == "#") {
                node->left = NULL;
            } else {
                auto l = new TreeNode(stoi(v[i]));
                node->left = l;
                q.push(l);
            }
            i++;

            if (v[i] == "#") {
                node->right = NULL;
            } else {
                auto l = new TreeNode(stoi(v[i]));
                node->right = l;
                q.push(l);
            }
            i++;
            q.pop();
        }

        return root;
    }

    vector<string> splitString(string& data)
    {
        vector<string> res;
        int id = 0;
        while (1) {
            int k = data.find(',', id);
            if (k == data.npos) break;
            res.push_back(data.substr(id, k - id));
            id = k + 1;
        }

        return res;
    }
};
```

### 方法2 DFS
```
class Codec {
public:
    string path;
    void dfs_s(TreeNode* root) {
        if (!root) {
            path += "#,";
        } else {
            path += to_string(root->val) + ",";
            dfs_s(root->left);
            dfs_s(root->right);
        }
    }

    // Encodes a tree to a single string.
    string serialize(TreeNode* root) {
        dfs_s(root);

        return path;
    }

    // Decodes your encoded data to tree.
    TreeNode* deserialize(string data) {
        // cout << "data: " << data << endl;
        int u = 0;
        return dfs_d(data, u);
    }

    TreeNode* dfs_d(string& data, int& u)
    {
        if (data[u] == '#') {
            u += 2;
            return NULL;
        } 

        int k = u;
        while (u < data.size() && data[u] != ',') u++;
        auto root = new TreeNode(stoi(data.substr(k, u - k)));
        u++;
        root->left = dfs_d(data, u);
        root->right = dfs_d(data, u);

        return root;
    }
};
```
# 402. 移掉K位数字
### 方法1 贪心 + 单调栈
```
class Solution {
public:
    string removeKdigits(string num, int k) {
        stack<char> st;
        for (auto x : num) {
            while (st.size() && st.top() > x && k) st.pop(), k--;
            st.push(x);
        }

        while (k--) st.pop();
        string res;
        while (st.size()) res += st.top(), st.pop();

        reverse(res.begin(), res.end());
        int i = 0;
        while (i < res.size() && res[i] == '0') i++;

        if (i == res.size()) return "0";
        return res.substr(i);
    }
};
```
### 方法2 贪心
```
class Solution {
public:
    string removeKdigits(string num, int k) {
        string res("0");
        for (auto x : num) {
            while (k && res.back() > x) res.pop_back(), k--;
            res += x;
        }
        while (k--) res.pop_back();
        k = 0;
        while (k + 1 < res.size() && res[k] == '0') k++;

        res = res.substr(k);

        return res;
    }
};
```
# 739. 每日温度
### 单调栈
```
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& temperatures) {
        int n = temperatures.size();
        vector<int> res(n, 0);
        stack<int> st;
        for (int i = 0; i < n; i++) {
            if (st.empty()) {
                st.push(i);
            } else {
                while (st.size() && temperatures[st.top()] < temperatures[i]) {
                    int x = st.top();
                    st.pop();
                    res[x] = i - x;
                }
                st.push(i);
            }
        }

        return res;
    }
};
```
# 207. 课程表
### 拓扑排序
```
class Solution {
public:
    bool canFinish(int n, vector<vector<int>>& p) {
        vector<int> inDegree(n);
        vector<int> ans;
        vector<vector<int>> g(n);
        for (auto x : p) {
            int a = x[0], b = x[1];
            inDegree[a]++;
            g[b].push_back(a);
        }

        queue<int> q;

        for (int i = 0; i < n; i++) {
            if (inDegree[i] == 0) {
                q.push(i);
            }
        }

        while (q.size()) {
            auto t = q.front();
            q.pop();
            ans.push_back(t);
            for (auto x : g[t]) {
                inDegree[x]--;
                if (inDegree[x] == 0) {
                    q.push(x);
                }
            }
        }

        return ans.size() == n;
    }
};
```
# 剑指 Offer 36. 二叉搜索树与双向链表
```
class Solution {
public:
    Node* pre;
    Node* treeToDoublyList(Node* root) {
        if (!root) return nullptr;
        pre = nullptr;
        dfs(root);
        auto l = root, r = root;
        while (l && l->left) {
            l = l->left;
        }

        while (r && r->right) {
            r = r->right;
        }
        l->left = r;
        r->right = l;

        return l;
    }

    void dfs(Node* root) {
        if (!root) return;
        dfs(root->left);
        if (pre) pre->right = root;
        root->left = pre;
        pre = root;
        dfs(root->right);
    }
};
```
# 958. 二叉树的完全性检验
### 层序遍历
```
class Solution {
public:
    bool isCompleteTree(TreeNode* root) {
        if (!root) return true;
        queue<TreeNode*> q;
        q.push(root);

        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                if (node == nullptr) {
                    while (q.size()) {
                        auto t = q.front();
                        q.pop();
                        if (t) return false;
                    }

                    return true;
                }
                q.push(node->left);
                q.push(node->right);
                if (!node->left && node->right) return false;
            }
        }

        return true;
    }
};
```
### 方法2 dfs
```
class Solution {
public:
    int n, p;

    bool isCompleteTree(TreeNode* root) {
        n = 0, p = 0;
        if (!dfs(root, 1)) return false;

        return n == p;
    }

    bool dfs(TreeNode* root, int k)
    {
        if (!root) return true;
        if (k > 100) return false;
        p = max(p, k);
        n++;
        return dfs(root->left, 2 * k) && dfs(root->right, 2 * k + 1);
    }
};
```


# 剑指 Offer 54. 二叉搜索树的第k大节点
### 方法1 中序遍历
```
class Solution {
public:
    vector<int> res;
    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->right);
        res.push_back(root->val);
        dfs(root->left);
    }
    int kthLargest(TreeNode* root, int k) {
        dfs(root);

        return res[k - 1];
    }
};
```

### 方法2 中序遍历
```
class Solution {
public:
    int res;
    void dfs(TreeNode* root, int& k) {
        if (!root) return;
        dfs(root->right, k);
        k--;
        if (k == 0) {
            res = root->val;
            return;
        }
        dfs(root->left, k);
    }

    int kthLargest(TreeNode* root, int k) {
        dfs(root, k);

        return res;
    }
};
```

# 79. 单词搜索
```
class Solution {
public:
    vector<vector<char>> g;
    vector<vector<bool>> st;
    int n, m;
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    string s;
    bool exist(vector<vector<char>>& board, string word) {
        g = board;
        s = word;
        n = g.size(), m = g[0].size();
        st = vector<vector<bool>>(n, vector<bool>(m, false));

        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                if (g[i][j] == word[0]) {
                    if (dfs(i, j, 0)) return true;
                }
            }
        }

        return false;
    }

    bool dfs(int x, int y, int u) {
        if (u == s.size() - 1) {
            return true;
        }

        st[x][y] = true;

        for (int i = 0; i < 4; i++) {
            int a = x + dx[i], b = y + dy[i];
            if (a >= 0 && a < n && b >= 0 && b < m && !st[a][b] && g[a][b] == s[u + 1]) {
                st[a][b] = true;
                if (dfs(a, b, u + 1)) return true;
                st[a][b] = false;
            }
        }

        st[x][y] = false;
        return false;
    }
};
```
# 11. 盛最多水的容器
### 双指针
```
class Solution {
public:
    int maxArea(vector<int>& h) {
        int l = 0, r = h.size() - 1, res = 0;
        while (l < r) {
            res = max(res, min(h[l], h[r]) * (r - l));
            if (h[l] < h[r]) l++;
            else r--;
        }

        return res;
    }
};
```
# 224. 基本计算器
```
class Solution {
public:
    int calculate(string s) {
        int n = s.size();
        int res = 0, sign = 1;
        stack<int> st;
        for (int i = 0; i < n; i++) {
            if (s[i] >= '0' && s[i] <= '9') {
                int j = i;
                int num = 0;
                while (j < n && s[j] >= '0' && s[j] <= '9') {
                    num = num * 10 - '0' + s[j];
                    j++;
                }

                res += num * sign;
                i = j - 1;
            } else if (s[i] == '+') {
                sign = 1;
            } else if (s[i] == '-') {
                sign = -1;
            } else if (s[i] == '(') {
                st.push(res);
                st.push(sign);
                res = 0;
                sign = 1;
            } else if (s[i] == ')') {
                res = res * st.top();
                st.pop();
                res += st.top();
                st.pop();
            }
        }

        return res;
    }
};
```
# 50. Pow(x, n)
```
class Solution {
public: 
    typedef long long LL;
    double myPow(double x, int n) {
        if (x == 0) return 0;
        if (n == 1) return x;
        if (x == 1 || n == 0) return 1;
        LL N = n;
        if (n < 0) x = 1 / x, N = - N;
        if (N % 2) return x * myPow(x * x, (N - 1) / 2);

        return myPow(x * x, N / 2);
    }
};
```
# 剑指 Offer 10- II. 青蛙跳台阶问题
```
const int N = 1E9 + 7;

class Solution {
public:
    int numWays(int n) {
        if (n == 0) return 1;
        if (n == 1) return 1;
        if (n == 2) return 2;
        int f0 = 1, f1 = 2;
        int res;
        for (int i = 3; i <= n; i++) {
            res = (f0 + f1) % N;
            f0 = f1;
            f1 = res;
        }

        return res;
    }
};
```
# 55. 跳跃游戏
### 方法1：贪心
```
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        int r = nums[0];
        for (int i = 1; i < n; i++) {
            if (r >= i) {
                r = max(r, i + nums[i]);
            } else {
                return false;
            }
        }
        return true;
    }
};
```
### 方法2：dp
```
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n);
        dp[0] = nums[0];
        for (int i = 1; i < n; i++) {
            if (dp[i - 1] < i) return false;
            dp[i] = max(dp[i - 1], i + nums[i]);
        }

        return dp[n - 1] >= n - 1;
    }
};
```
# 45. 跳跃游戏 II
## 还有遍历做法未实现
### codetop 方法1：dp 自己想出来的
```
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 0x3f3f3f3f);
        dp[0] = 0;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[j] + j >= i)
                    dp[i] = min(dp[i], dp[j]);
            }
            dp[i] += 1;
        }

        return dp[n - 1];
    }
};
```

### codetop 方法2：dp 
```
class Solution {
public:
    int jump(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n);
        for (int i = 1, j = 0; i < n; i++) {
            while (j + nums[j] < i) j++;
            dp[i] = dp[j] + 1;
        }

        return dp[n - 1];
    }
};
```
# 59. 螺旋矩阵 II
```
class Solution {
public:
    vector<vector<int>> generateMatrix(int n) {
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};
        int d = 1;
        vector<vector<bool>> st(n, vector<bool>(n));
        vector<vector<int>> res(n, vector<int>(n));
        int x = 0, y = 0;
        for (int i = 0; i < n * n; i++) {
            res[x][y] = i + 1;
            int a = x + dx[d], b = y + dy[d];
            st[x][y] = true;
            if (a >= 0 && a < n && b >= 0 && b < n && !st[a][b]) {
                x = a, y = b;
            } else {
                d = (d + 1) % 4;
                x = x + dx[d];
                y = y + dy[d];
            }
        }

        return res;
    }
};
```
# 74. 搜索二维矩阵
### 双指针
```
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && i >= 0 && j >= 0 && j < n) {
            if (matrix[i][j] > target) j--;
            else if (matrix[i][j] < target) i++;
            else return true;
        }

        return false;
    }
};
```

# 26. 删除排序数组中的重复项
```
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if (nums.empty()) return 0;
        int k = 0;
        nums[k] = nums[0];
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] == nums[k]) continue;
            else nums[++k] = nums[i];
        }

        return k + 1;
    }
};
```
# 剑指 Offer 21. 调整数组顺序使奇数位于偶数前面
### 方法1
```
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int n = nums.size();
        int l = 0, r = n - 1;
        vector<int> res(n);
        for (int i = 0; i < n; i++) {
            if (nums[i] % 2) res[l++] = nums[i];
            else res[r--] = nums[i];
        }

        return res;
    }
};
```
### 方法2
```
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int n = nums.size();
        int l = 0, r = n - 1;
        // vector<int> res(n);
        while (l < r) {
            while (l < n && nums[l] % 2) l++;
            while (r >= 0 && nums[r] % 2 == 0) r--;
            if (l < r) swap(nums[l], nums[r]);
        }

        return nums;
    }
};
```

# 7. 整数反转
### 方法1
```
class Solution {
public:
    int reverse(int x) {
        int n = 0;
        while (x > 0) {
            int t = x % 10; 
            x /= 10;
            if (n > (INT_MAX - t) / 10) return 0;
            n = n * 10 + t;
        }

        while (x < 0) {
            int t = x % 10;
            x /= 10;
            cout << t << " " << x << endl;
            if (n < (INT_MIN - t) / 10) return 0;
            n = n * 10 + t;
        }

        return n;
    }
};
```
### 方法2
```
class Solution {
public:
    int reverse(int x) {
        typedef long long LL;
        int f = 1;
        LL xx = x;
        if (xx < 0) xx = -xx, f = -1;
        string s = to_string(xx);
        std::reverse(s.begin(), s.end());
        LL n = stol(s);
        if (n >= INT_MAX || n < INT_MIN) return 0;

        return n * f;
    }
};
```
# 剑指 Offer 40. 最小的k个数
### 方法1
```
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        sort(arr.begin(), arr.end());
        vector<int> ans;

        for (int i = 0; i < k; i++) {
            ans.push_back(arr[i]);
        }

        return ans;
    }
};
```
### 方法2
```
class Solution {
public:
    vector<int> getLeastNumbers(vector<int>& arr, int k) {
        priority_queue<int, vector<int>, greater<int>> h;
        vector<int> ans;

        for (int i = 0; i < arr.size(); i++) {
            h.push(arr[i]);
            if (h.size() > arr.size() - k) {
                ans.push_back(h.top());
                h.pop();
            }
        }

        return ans;
    }
};
```

# 剑指 Offer 10- I. 斐波那契数列
```
class Solution {
public:
    int fib(int n) {
        if (n <= 1) return n;
        int f0 = 0, f1 = 1;
        int f = 0;
        for (int i = 2; i <= n; i++) {
            f = (f0 + f1) % 1000000007;
            f0 = f1;
            f1 = f;
        }

        return f;
    }
};
```

# 剑指 Offer 42. 连续子数组的最大和
### 方法1：dp
```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int res = nums[0];
        vector<int> dp(n, -1e9);
        dp[0] = nums[0];
        for (int i = 1; i < n; i++) {
           dp[i] = max(dp[i - 1] + nums[i], nums[i]);
           res = max(res, dp[i]);
        }

        return res;
    }
};
```
### 方法2：单次循环
```
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int n = nums.size();
        int res = INT_MIN;

        for (int i = 0, last = 0; i < n; i++) {
           last = max(last + nums[i], nums[i]);
           res = max(res, last);
        }

        return res;
    }
};
```

# 518. 零钱兑换 II
### 方法1：朴素背包
```
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        sort(coins.begin(), coins.end());
        vector<vector<int>> dp(n + 1, vector<int>(amount + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= amount; j++) {
                // dp[i][j] += dp[i - 1][j];
                for (int k = 0; k * coins[i - 1] <= j; k++) {
                    dp[i][j] += dp[i - 1][j - k * coins[i - 1]];
                }
            }
        }

        return dp[n][amount];
    }
};
```
### 方法2：背包优化
```
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        sort(coins.begin(), coins.end());
        vector<vector<int>> dp(n + 1, vector<int>(amount + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            dp[i][0] = 1;
            for (int j = 1; j <= amount; j++) {
                dp[i][j] += dp[i - 1][j];
                if (j >= coins[i - 1]) {
                    dp[i][j] += dp[i][j - coins[i - 1]];
                }
            }
        }

        return dp[n][amount];
    }
};
```
### 方法3：优化成一维
```
class Solution {
public:
    int change(int amount, vector<int>& coins) {
        int n = coins.size();
        sort(coins.begin(), coins.end());
        // vector<vector<int>> dp(n + 1, vector<int>(amount + 1));
        vector<int> dp(amount + 1);
        // dp[0][0] = 1;
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= amount; j++) {
                // dp[i][j] += dp[i - 1][j];
                if (j >= coins[i - 1]) {
                    dp[j] += dp[j - coins[i - 1]];
                    // dp[i][j] += dp[i][j - coins[i - 1]];
                }
            }
        }

        return dp[amount];
    }
};
```
# 145. 二叉树的后序遍历
### 方法1：递归
```
class Solution {
public:
    vector<int> ans;
    
    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        dfs(root->right);
        ans.push_back(root->val);
    }

    vector<int> postorderTraversal(TreeNode* root) {
        dfs(root);

        return ans;
    }
};
```
### 方法2：迭代算法，前序遍历的变形
```
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        stack<TreeNode*> st;

        while (st.size() || root) {
            if (root) {
                ans.push_back(root->val);
                st.push(root);
                root = root->right;
            } else {
                root = st.top()->left;
                st.pop();
            }
        }

        reverse(ans.begin(), ans.end());

        return ans;
    }
};
```
### 方法3：迭代算法后序遍历
```
class Solution {
public:
    vector<int> postorderTraversal(TreeNode* root) {
        vector<int> ans;
        TreeNode* pre = nullptr;
        stack<TreeNode*> st;
        while (st.size() || root) {
            while (root) {
                st.push(root);
                root = root->left;
            }

            root = st.top();
            st.pop();

            if (root->right == nullptr || root->right == pre) {
                ans.push_back(root->val);
                pre = root;
                root = nullptr;
            } else {
                st.push(root);
                root = root->right;
            }
        }

        return ans;
    }
};
```
# 40. 组合总和 II
### dfs去重是重点
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<bool> st;
    int n;
    vector<vector<int>> combinationSum2(vector<int>& num, int target) {
        sort(num.begin(), num.end());
        n = num.size();
        st = vector<bool>(n, false);
        vector<int> path;
        dfs(num, target, path, 0, 0);

        return ans;
    }

    void dfs(vector<int>& num, int target, vector<int> path, int id, int s)
    {
        // if (id >= n) return;
        if (s > target) return;
        if (s == target) {
            ans.push_back(path);
            return;
        }
        
        for (int i = id; i < n; i++) {
            if (!st[i]) {
                if (i != id && num[i] == num[i - 1]) continue;
                st[i] = true;
                path.push_back(num[i]);
                dfs(num, target, path, i + 1, s + num[i]);
                st[i] = false;
                path.pop_back();
            }
        }
    }
};
```
# 补充题5. 手撕归并排序
### 归并排序
```
class Solution {
public:
    vector<int> help;
    void mergeSort(vector<int>& nums, int l, int r) {
        if (l >= r) return; // 注意必须写等于号
        int mid = (l + r) >> 1;
        mergeSort(nums, l, mid);
        mergeSort(nums, mid + 1, r);
        int k = 0, i = l, j = mid + 1;
        while (i <= mid && j <= r) {
            if (nums[i] <= nums[j]) help[k++] = nums[i++];
            else help[k++] = nums[j++];
        }

        while (i <= mid) help[k++] = nums[i++];
        while (j <= r) help[k++] = nums[j++];
        for (int i = l, j = 0; i <= r; i++, j++) {
            nums[i] = help[j];
        }
    }


    vector<int> sortArray(vector<int>& nums) {
        help = nums;
        mergeSort(nums, 0, nums.size() - 1);

        return nums;
    }
};
```

# 补充题1. 排序奇升偶降链表
### 链表拆分 + 反转链表
```牛客上有
/**
 * struct ListNode {
 *	int val;
 *	struct ListNode *next;
 *	ListNode(int x) : val(x), next(nullptr) {}
 * };
 */
class Solution {
public:
    /**
     * 代码中的类名、方法名、参数名已经指定，请勿修改，直接返回方法规定的值即可
     *
     * 
     * @param head ListNode类 
     * @return ListNode类
     */
    ListNode* sortLinkedList(ListNode* head) {
        auto p0 = new ListNode(-1);
        auto p1 = new ListNode(-2);
        auto n0 = p0, n1 = p1;
        int i = 1;
        while (head) {
            if (i % 2) {
                n0->next = head;
                n0 = n0->next;
            } else {
                n1->next = head;
                n1 = n1->next;
            }

            head = head->next;
            i++;
        }

        cout << head << " " << i << endl;
        n0->next = nullptr;
        n1->next = nullptr;
        n0 = p0->next, n1 = p1->next;
        p1->next = nullptr;
        while (n1) {
            auto t = n1->next;
            n1->next = p1->next;
            p1->next = n1;
            n1 = t;
        }
        n1 = p1->next;
        auto dummy = new ListNode(0);
        auto p = dummy;

        while (n0 && n1) {
            if (n0->val < n1->val) {
                p->next = n0;
                p = p->next;
                n0 = n0->next;
            } else {
                p->next = n1;
                p = p->next;
                n1 = n1->next;
            }
        }

        if (n0) p->next = n0;
        if (n1) p->next = n1;

        return dummy->next;
    }
};
```

# 补充题23. 检测循环依赖
### 拓扑排序
```
class Solution {
public:
    bool canFinish(int n, vector<vector<int>>& p) {
        vector<int> ans;
        vector<int> ind(n, 0);
        vector<vector<int>> g(n);
        for (auto x : p) {
            int a = x[0], b = x[1];
            g[b].push_back(a);
            ind[a]++;
        }

        queue<int> q;

        for (int i = 0; i < n; i++) {
            if (ind[i] == 0) {
                q.push(i);
            }
        }

        while (q.size()) {
            int val = q.front();
            q.pop();
            ans.push_back(val);
            for (auto x : g[val]) {
                ind[x]--;
                if (ind[x] == 0) q.push(x);
            }
        }

        return ans.size() == n;
    }
};
```
# 剑指 Offer 51. 数组中的逆序对
```
class Solution {
public:
    typedef long long LL;
    int n;
    vector<int> a, help;
    int reversePairs(vector<int>& nums) {
        n = nums.size();
        a = nums;
        help = nums;
        return mergeSort(0, nums.size() - 1);
    }

    LL mergeSort(int l, int r) {
        if (l >= r) return 0;

        LL mid = (l + r) >> 1;
        LL res = mergeSort(l, mid) + mergeSort(mid + 1, r);

        int i = l, j = mid + 1, k = 0;
        while (i <= mid && j <= r) {
            if (a[i] <= a[j]) {
                help[k++] = a[i++];
            } else {
                help[k++] = a[j++];
                res += (mid - i + 1);
            }
        }

        while (i <= mid) help[k++] = a[i++];
        while (j <= r) help[k++] = a[j++];
        for (int i = l, j = 0; i <= r; i++, j++) {
            a[i] = help[j];
        }

        return res;
    }

};
```

# 61. 旋转链表
```
class Solution {
public:
    ListNode* rotateRight(ListNode* head, int k) {
        if (!head) return head;
        int n = 0;
        auto p = head;
        while (p) {
            p = p->next;
            n++;
        }

        k = k % n;
        if (k == 0) return head;
        k = n - k;
        p = head;
        while (p->next) {
            p = p->next;
        }
        p->next = head;
        ListNode* pre = nullptr;
        p = head;
        for (int i = 0; i < k; i++) {
            pre = p;
            p = p->next;
        }

        pre->next = nullptr;

        return p;

    }
};
```
# 450. 删除二叉搜索树中的节点
### 方法1
```
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        del(root, key);

        return root;
    }

    void del(TreeNode* &root, int key) {
        if (!root) return;

        if (root->val > key) del(root->left, key);
        else if (root->val < key) del(root->right, key);
        else {
            if (!root->left && !root->right) root = nullptr;
            else if (!root->left) root = root->right;
            else if (!root->right) root = root->left;
            else {
                auto p = root->right;
                while (p->left) {
                    p = p->left;
                }

                root->val = p->val;
                del(root->right, p->val);
            }
        }
    }
};
```
### 方法2
```
class Solution {
public:
    TreeNode* deleteNode(TreeNode* root, int key) {
        del(root, key);

        return root;
    }

    void del(TreeNode* &root, int key) {
        if (!root) return;

        if (root->val > key) del(root->left, key);
        else if (root->val < key) del(root->right, key);
        else {
            if (!root->left && !root->right) root = nullptr;
            else if (!root->left) root = root->right;
            else if (!root->right) root = root->left;
            else {
                auto p = root->left;
                while (p->right) {
                    p = p->right;
                }

                root->val = p->val;
                del(root->left, p->val);
            }
        }
    }
};
```
# 123. 买卖股票的最佳时机 III
### 两次遍历
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n <= 1) return 0;
        vector<int> f(n), g(n);
        for (int i = 1, last = prices[0], s = 0; i < n; i++) {
            s = max(s, prices[i] - last);
            f[i] = s;
            last = min(last, prices[i]);
            // cout << i << " " << f[i] << endl;
        }
        // cout << "------" << endl;
        for (int i = n - 2, last = prices[n - 1], s = 0; i >= 0; i--) {
            s = max(s, last - prices[i]);
            g[i] = s;
            last = max(last, prices[i]);
            // cout << i << " " << g[i] << endl;
        }

        int res = max(0, f[n - 1]);

        for (int i = 0; i < n - 1; i++) {
            res = max(f[i] + g[i + 1], res);
            // cout << "i: " << i << ", res : " << res << endl;
        }

        return res;
    }
};
```
# 235. 二叉搜索树的最近公共祖先
```
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {

        while (1) {
            if (root->val < p->val && root->val < q->val) {
                root = root->right;
            } else if (root->val > p->val && root->val > q->val) {
                root = root->left;
            } else return root;
        }

        return nullptr;
    }
};
```
### 方法2：递归
```
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root || root == p || root == q) return root;

        auto l = lowestCommonAncestor(root->left, p, q);
        auto r = lowestCommonAncestor(root->right, p, q);

        if (l && r) return root;
        else if (!l && r) return r;
        else if (l && !r) return l;

        return NULL;
    }
};
```


### 方法3
```
class Solution {
public:
    TreeNode* ans = nullptr;

    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        dfs(root, p, q);

        return ans;
    }

    int dfs(TreeNode* root, TreeNode* p, TreeNode* q) {
        if (!root) return 0;
        // if (root == p) return 1;
        // else if (root == q) return 2;
        int state = dfs(root->left, p, q);
        if (root == p) state |= 1;
        else if (root == q) state |= 2;
        state |= dfs(root->right, p, q);
        if (state == 3 && ans == nullptr) {
            ans = root;
            return 0;
        }

        return state;
    }
};
```

# 230. 二叉搜索树中第K小的元素
### 方法1
```
class Solution {
public:
    vector<int> v;
    int kthSmallest(TreeNode* root, int k) {
        dfs(root);

        return v[k - 1];
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        v.push_back(root->val);
        dfs(root->right);
    }
};
```
### 方法2
```
class Solution {
public:
    int ans;
    int kthSmallest(TreeNode* root, int k) {
        dfs(root, k);

        return ans;
    }

    void dfs(TreeNode* root, int &k) {
        if (!root) return;
        dfs(root->left, k);
        k--;
        if (k == 0) {
            ans = root->val;
            return;
        }
        dfs(root->right, k);
    }
};
```
# 75. 颜色分类
### 方法1：两次遍历
```
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        vector<int> count(3, 0);
        for (int i = 0; i < n; i++)
            count[nums[i]]++;

        for (int i = 0, id = 0; i < 3; i++) {
            for (int j = 0;  j < count[i]; j++) {
                nums[id++] = i;
            }
        }
    }
};
```
### 方法2：单次遍历，双指针
```
class Solution {
public:
    void sortColors(vector<int>& nums) {
        int n = nums.size();
        int l = 0, r = n - 1;
        for (int i = 0; i <= r ; ) {
            if (nums[i] == 0) swap(nums[l++], nums[i++]);
            else if (nums[i] == 2) swap(nums[r--], nums[i]);
            else i++;
        }
    }
};
```
### 方法3：快排的partion
```
class Solution {
public:
    void sortColors(vector<int>& nums) {
        partion(nums, 1);
    }

    void partion(vector<int>& nums, int p) {
        int l = -1, r = nums.size(), id = 0;
        while (id < r) {
            if (nums[id] > p) {
                swap(nums[--r], nums[id]);
            } else if (nums[id] < p) {
                swap(nums[++l], nums[id++]);
            } else {
                id++;
            }
        }
    }
};
```

# 135. 分发糖果
```
class Solution {
public:
    int candy(vector<int>& ratings) {
        int n = ratings.size();
        vector<int> a(n, 1);
        for (int i = 1; i < n; i++) {
            if (ratings[i] > ratings[i - 1]) {
                a[i] = a[i - 1] + 1;
            }
        }

        for (int i = n - 2; i >= 0; i--) {
            if (ratings[i] > ratings[i + 1]) {
                a[i] = max(a[i], a[i + 1] + 1);
            }
        }

        int res = 0;

        for (auto x : a) res += x;

        return res;
    }
};
```
# 剑指 Offer 62. 圆圈中最后剩下的数字
### 方法1：递归
```
class Solution {
public:
    int lastRemaining(int n, int m) {
        if (n == 1) return 0;

        return (lastRemaining(n - 1, m) + m) % n;
    }
};
```
### 方法2：迭代
```
class Solution {
public:
    int lastRemaining(int n, int m) {
        if (n == 1) return 0;
        int f = 0;
        for (int i = 2; i <= n; i++) 
            f = (f + m) % i;
        return f;
    }
};
```

# 补充题2. 圆环回原点问题
### 动态规划
```
// https://www.nowcoder.com/questionTerminal/16409dd00ab24a408ddd0c46e49ddcf8
class Solution {
public:
    int circle(int n) {
        // write code here
        const int N = 1e9 + 7;
        const int NO = 10;
        vector<vector<int>> dp(n + 1, vector<int>(NO + 1));
        dp[0][0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j < NO; j++) {
                dp[i][j] = (dp[i - 1][(j - 1 + NO) % NO] % N  + dp[i - 1][(j + 1 + NO) % NO] % N) % N;
                //cout << i << " " << j << " " << dp[i][j] << endl;
            }
        }
        
        return dp[n][0];
    }
};
```

# 225. 用队列实现栈
```
class MyStack {
public:
    queue<int> q0, q1;

    MyStack() {

    }
    
    void push(int x) {
        q0.push(x);
    }
    
    int pop() {
        while (q0.size() > 1) {
            q1.push(q0.front());
            q0.pop();
        }

        int res = q0.front();
        q0.pop();
        while (q1.size()) {
            q0.push(q1.front());
            q1.pop();
        }
        return res;
    }
    
    int top() {
        while (q0.size() > 1) {
            q1.push(q0.front());
            q0.pop();
        }

        int res = q0.front();
        q1.push(res);
        q0.pop();
        while (q1.size()) {
            q0.push(q1.front());
            q1.pop();
        }
        return res;
    }
    
    bool empty() {
        return q0.empty();
    }
};
```
# 剑指 Offer 04. 二维数组中的查找
```
class Solution {
public:
    bool findNumberIn2DArray(vector<vector<int>>& matrix, int target) {
        if (matrix.empty()) return 0;
        int n = matrix.size(), m = matrix[0].size();
        int i = 0, j = m - 1;
        while (i < n && j >= 0) {
            if (matrix[i][j] > target) j--;
            else if (matrix[i][j] < target) i++;
            else return true;
        }

        return false;
    }
};
```
# 440. 字典序的第K小数字
### 方法1
```
class Solution {
public:
    typedef long long LL;
    int findKthNumber(int n, int k) {
        int prefix = 1;
        while (k > 1) {
            int cnt = getNum(prefix, n);
            if (k > cnt) {
                k -= cnt;
                prefix++;
            } else {
                k--;
                prefix *= 10;
            }
        }

        return prefix;
    }

    int getNum(int prefix, int n) {
        int ans = 0;
        LL t = prefix, k = 1;
        while (t * 10 <= n) {
            ans += k;
            k *= 10;
            t *= 10;
        }

        if (n - t < k) ans += n - t + 1;
        else ans += k;

        return ans;
    }
};
```
### 方法2
```
class Solution {
public:
    typedef long long LL;
    int findKthNumber(int n, int k) {
        int prefix = 1;
        while (k > 1) {
            int cnt = getNum(prefix, n);
            if (k > cnt) {
                k -= cnt;
                prefix++;
            } else {
                k--;
                prefix *= 10;
            }
        }

        return prefix;
    }

    int getNum(int prefix, int n) {
        long long p = 1;
        auto A = to_string(n), B = to_string(prefix);
        int dt = A.size() - B.size();
        int res = 0;
        for (int i = 0; i < dt; i ++ ) {
            res += p;
            p *= 10;
        }
        A = A.substr(0, B.size());
        if (A == B) res += n - prefix * p + 1;
        else if (A > B) res += p;
        return res;
    }
};
```
# 328. 奇偶链表
```
class Solution {
public:
    ListNode* oddEvenList(ListNode* head) {
        auto p0 = new ListNode(), p1 = new ListNode();
        auto node0 = p0, node1 = p1;
        int n = 1;
        auto p = head;
        while (p) {
            if (n % 2) {
                p0->next = p;
                p0 = p0->next;
                p = p->next;
            } else {
                p1->next = p;
                p1 = p1->next;
                p = p->next;
            }
            n++;
        }
        p1->next = nullptr;
        p0->next = node1->next;

        return node0->next;
    }
};
```
# 91. 解码方法
```
class Solution {
public:
    int numDecodings(string s) {
        int n = s.size();
        s = " " + s;
        vector<int> dp(n + 1);
        dp[0] = 1;
        for (int i = 1; i <= n; i++) {
            if (s[i] >= '1' && s[i] <= '9') {
                dp[i] = dp[i - 1];
            }

            if (i > 1) {
                int t = 10 * (s[i - 1] - '0') + s[i] - '0';
                if (t >= 10 && t <= 26) {
                    dp[i] += dp[i - 2];
                }
            }
        }

        return dp[n];
    }
};
```

# 572. 另一棵树的子树
```
class Solution {
public:
    bool isSubtree(TreeNode* root, TreeNode* subRoot) {
        return dfs(root, subRoot);
    }

    bool isSame(TreeNode* root, TreeNode* subRoot) {
        if (!root && !subRoot) return true;
        if (!root && subRoot) return false;
        if (root && !subRoot) return false;
        if (root->val != subRoot->val) return false;

        auto l = isSame(root->left, subRoot->left);
        auto r = isSame(root->right, subRoot->right);

        if (l && r) return true;

        return false;
    }

    bool dfs(TreeNode* root, TreeNode* subRoot) {
        if (isSame(root, subRoot)) return true;

        if (root && dfs(root->left, subRoot)) return true;
        if (root && dfs(root->right, subRoot)) return true;

        return false;

    }
};
```
# 329. 矩阵中的最长递增路径
### 记忆化搜索
```
class Solution {
public:
    int dx[4] = {-1, 0, 1, 0};
    int dy[4] = {0, 1, 0, -1};
    vector<vector<int>> f, w;
    int n, m;
    
    int longestIncreasingPath(vector<vector<int>>& matrix) {
        w = matrix;
        n = matrix.size(), m = matrix[0].size();
        f = vector<vector<int>>(n, vector<int>(m, -1));
        int res = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < m; j++) {
                res = max(res, dp(i, j));
                // cout << i << " " << j << " " << dp(i, j) << endl;
            }
        }

        return res;
    }

    int dp(int a, int b) {
        if (f[a][b] != -1) return f[a][b];
        f[a][b] = 1;

        for (int i = 0; i < 4; i++) {
            int x = a + dx[i], y = b + dy[i];
            if (x >= 0 && x < n && y >= 0 && y < m && w[x][y] > w[a][b]) {
                f[a][b] = max(f[a][b], dp(x, y) + 1);
            }
        }

        return f[a][b];
    }
};
```
# 384. 打乱数组
### 洗牌算法
```
class Solution {
public:
    vector<int> oldArr, newArr;
    Solution(vector<int>& nums) {
        oldArr = newArr = nums;
    }
    
    vector<int> reset() {
        return oldArr;
    }
    
    vector<int> shuffle() {
        for (int i = 0; i < oldArr.size(); i++) {
            swap(newArr[i], newArr[i + rand() % (oldArr.size() - i)]);
        }

        return newArr;
    }
};
```
# 189. 轮转数组
### 两次反转
```
class Solution {
public:
    void rotate(vector<int>& nums, int k) {
        int n = nums.size();
        k = k % n;
        reverse(nums.begin(), nums.begin() + n - k);
        reverse(nums.begin() + n - k, nums.end());
        reverse(nums.begin(), nums.end());
    }
};
```
# 125. 验证回文串
### 双指针
```
class Solution {
public:
    bool isPalindrome(string s) {
        string str;
        for (auto x : s) {
            if (isalpha(x)) {
                str += tolower(x);
            } else if (isdigit(x)) {
                str += x;
            }
        }
        // cout << "str: " << str << endl;
        int l = 0, r = str.size() - 1;
        while (l < r) {
            if (str[l] != str[r]) return false;
            l++;
            r--;
        }

        return true;
    }
};
```
# 9. 回文数
### 方法1：转换为字符串
```
class Solution {
public:
    bool isPalindrome(int x) {
        string s = to_string(x);
        int i = 0, j = s.size() - 1;
        while (i < j) {
            if (s[i] != s[j]) return false;
            else i++, j--;
        }

        return true;
    }
};
```
### 方法2：数学
```
class Solution {
public:
    bool isPalindrome(int x) {
        if (x < 0) return false;
        typedef long long LL;
        LL num = 0;
        int n = x;
        while (n) {
            int t = n % 10;
            n /= 10;
            num = num * 10 + t;
        }

        if (num == x) return true;

        return false;
    }
};
```
# 445. 两数相加 II
### 方法1：反转链表后再处理
```
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        l2 = reverseList(l2);
        l1 = reverseList(l1);
        int c = 0;
        auto dummy = new ListNode();
        auto p = dummy;
        while (l1 && l2) {
            int t = l1->val + l2->val + c;
            c = t / 10;
            t %= 10;
            auto node = new ListNode(t);
            p->next = node;
            p = node;
            l1 = l1->next;
            l2 = l2->next;
        }

        while (l1) {
            int t = l1->val + c;
            c = t / 10;
            t %= 10;
            auto node = new ListNode(t);
            p->next = node;
            p = node;
            l1 = l1->next;
        }

        while (l2) {
            int t = l2->val + c;
            c = t / 10;
            t %= 10;
            auto node = new ListNode(t);
            p->next = node;
            p = node;
            l2 = l2->next;
        }

        if (c) {
            auto node = new ListNode(c);
            p->next = node;
            p = node;
        }

        return reverseList(dummy->next);
    }

    ListNode* reverseList(ListNode* p) {
        if (p == nullptr || p->next == nullptr) return p;
        auto a = p, b = p->next;
        while (b) {
            auto c = b->next;
            b->next = a;
            a = b;
            b = c;
        }
        p->next = nullptr;
        return a;
    }
};
```
### 方法2：栈
```
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        stack<int> st1, st2, st;
        while (l1) {
            st1.push(l1->val);
            l1 = l1->next;
        }

        while (l2) {
            st2.push(l2->val);
            l2 = l2->next;
        }

        int c = 0;
        while (st1.size() && st2.size()) {
            int t = st1.top() + st2.top() + c;
            c = t / 10;
            st.push(t % 10);
            st1.pop();
            st2.pop();
        }

        while (st1.size()) {
            int t = st1.top() + c;
            c = t / 10;
            st.push(t % 10);
            st1.pop();
        }

        while (st2.size()) {
            int t = st2.top() + c;
            c = t / 10;
            st.push(t % 10);
            st2.pop();
        }

        if (c) st.push(c);
        auto dummy = new ListNode();
        auto p = dummy;
        while (st.size()) {
            auto node = new ListNode(st.top());
            st.pop();
            p->next = node;
            p = node;
        }

        return dummy->next;
    }
};
```

# 剑指 Offer 27. 二叉树的镜像
### 方法1：dfs
```
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if (!root) return NULL;
        TreeNode* l = mirrorTree(root->left);
        TreeNode* r = mirrorTree(root->right);
        root->left = r;
        root->right = l;

        return root;
    }
};
```
### 方法2：dfs
```
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        dfs(root);

        return root;
    }

    void dfs(TreeNode* root)
    {
        if (!root) return;
        if (root->left == NULL && root->right == NULL) return;
        auto t = root->left;
        root->left = root->right;
        root->right = t;
        dfs(root->left);
        dfs(root->right);
    }
};
```
### 方法3：bfs
```
class Solution {
public:
    TreeNode* mirrorTree(TreeNode* root) {
        if (!root) return root;
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                auto left = node->left;
                auto right = node->right;
                node->left = right;
                node->right = left;
                if (left) q.push(left);
                if (right) q.push(right);
            }
        }

        return root;
    }
};
```

# 287. 寻找重复数
### 二分数据范围
```
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int l = 1, r = nums.size() - 1;
        while (l < r) {
            int mid = l + r >> 1;
            int cnt = 0;
            for (auto x : nums) {
                if (x >= l && x <= mid) cnt++;
            }

            if (cnt > mid - l + 1) r = mid;
            else l = mid + 1;
        }

        return l;
    }
};
```
### 方法2：快慢指针
```
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int s = 0, f = 0;
        do {
            s = nums[s];
            f = nums[nums[f]];
        } while (s != f);

        f = 0;
        do {
            s = nums[s];
            f = nums[f];
        } while (s != f);

        return s;
    }
};
```
# 208. 实现 Trie (前缀树)
### 方法1
```
const int N = 3E5 + 10;
class Trie {
public:
    int son[N][26], cnt[N], idx;
    Trie() {
        memset(son, 0, sizeof son);
        memset(cnt, 0, sizeof cnt);
        idx = 0;
    }
    
    void insert(string word) {
        int p = 0;
        for (auto& x: word) {
            int u = x - 'a';
            if (!son[p][u]) son[p][u] = ++idx;
            p = son[p][u];
        }
        cnt[p]++;
    }
    
    bool search(string word) {
        int p = 0;

        for (auto& x: word) {
            int u = x - 'a';
            if (!son[p][u]) return false;
            p = son[p][u];
        }

        return cnt[p];
    }
    
    bool startsWith(string prefix) {
        int p = 0;

        for (auto& x: prefix) {
            int u = x - 'a';
            if (!son[p][u]) return false;
            p = son[p][u];
        }

        return true;
    }
};
```
### 方法2
```
class Trie {
public:
    struct Node {
        Node* son[26];
        bool isEnd;
        Node() : isEnd(false) {
            // memset(son, 0, sizeof son);
            for (int i = 0; i < 26; i++) son[i] = nullptr;
        }
    };

    Node* root;

    Trie() {
        root = new Node();
    }
    
    void insert(string word) {
        auto p = root;
        for (auto x : word) {
            int t = x - 'a';
            if (!p->son[t]) p->son[t] = new Node();
            p = p->son[t];
        }
        p->isEnd = true;
    }
    
    bool search(string word) {
        auto p = root;
        for (auto x : word) {
            int t = x - 'a';
            if (!p->son[t]) return false;
            p = p->son[t];
        }

        return p->isEnd;
    }
    
    bool startsWith(string prefix) {
        auto p = root;
        for (auto x : prefix) {
            int t = x - 'a';
            if (!p->son[t]) return false;
            p = p->son[t];
        }

        return true;
    }
};
```

# 295. 数据流的中位数
### 方法1：两次while 保证下面比上面多
```
class MedianFinder {
public:
    priority_queue<int> down; // 大根堆
    priority_queue<int, vector<int>, greater<int>> up; //小根堆

    MedianFinder() {

    }
    
    void addNum(int num) {
        if (down.empty())
            down.push(num);
        else {
            if (num > down.top()) {
                up.push(num);
            } else {
                down.push(num);
            }
        }

        while (up.size() >= down.size()) {
            down.push(up.top());
            up.pop();
        }

        while (down.size() > up.size() + 1) {
            up.push(down.top());
            down.pop();
        }
    }
    
    double findMedian() {
        if (up.size() == down.size()) {
            int a = up.top(), b = down.top();

            return (a + b) / 2.0;
        } 

        return down.top();
    }
};
```
### 方法2：每次都插入下面，保证下面的堆比上面的多
```
class MedianFinder {
public:
    priority_queue<int> down; // 大根堆
    priority_queue<int, vector<int>, greater<int>> up; //小根堆

    MedianFinder() {

    }
    
    void addNum(int num) {
        if (up.empty())
            down.push(num);
        else {
            if (num > down.top()) {
                up.push(num);
                down.push(up.top());
                up.pop();
            } else {
                down.push(num);
            }
        }

        // while (up.size() >= down.size()) {
        //     down.push(up.top());
        //     up.pop();
        // }

        if (down.size() > up.size() + 1) {
            up.push(down.top());
            down.pop();
        }
    }
    
    double findMedian() {
        if (up.size() == down.size()) {
            int a = up.top(), b = down.top();

            return (a + b) / 2.0;
        } 

        return down.top();
    }
};
```

# 114. 二叉树展开为链表
### 方法1：暴力
```
class Solution {
public:
    vector<TreeNode*> ans;

    void dfs(TreeNode* root) {
        if (!root) return;
        ans.push_back(root);
        dfs(root->left);
        dfs(root->right);
    }
    void flatten(TreeNode* root) {
        dfs(root);
        auto p = root;
        for (int i = 1; i < ans.size(); i++) {
            p->left = nullptr;
            p->right = ans[i];
            p = p->right;
        }
    }
};
```
### 方法2：前序遍历的特点
```
class Solution {
public:
    void flatten(TreeNode* root) {
        while (root) {
            auto p = root->left;
            if (p) { 
                while (p->right) p = p->right;
                p->right = root->right;
                root->right = root->left;
                root->left = nullptr;
            }
            root = root->right;
        }
    }
};
```
# 剑指 Offer 26. 树的子结构
### 方法1：注意和572题比较
```
class Solution {
public:
    bool isSubStructure(TreeNode* a, TreeNode* b) {
        bool r = false;
        if (!a || !b) return false;

        if (dfs(a, b)) return true;

        return isSubStructure(a->left, b) || isSubStructure(a->right, b);
    }

    bool dfs(TreeNode* a, TreeNode* b) {
        if (b == NULL) return true;
        if (a == NULL) return false;

        if (a->val != b->val) return false;

        return dfs(a->left, b->left) && dfs(a->right, b->right);
    }
};
```
# 96. 不同的二叉搜索树
### 卡特兰数
```
class Solution {
public:
    int numTrees(int n) {
        vector<int> f(n + 1);
        f[0] = 1;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i; j++) {
                f[i] += f[j - 1] * f[i - j];
            }
        }

        return f[n];
    }
};
```
# 120. 三角形最小路径和
### dp
```
class Solution {
public:
    int minimumTotal(vector<vector<int>>& triangle) {
        int n = triangle.size(), m = triangle[0].size();
        vector<vector<int>> dp(n, vector<int>(n));
        dp[0][0] = triangle[0][0];
        int res = INT_MAX;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < triangle[i].size(); j++) {
                if (j == 0) dp[i][j] = dp[i - 1][j] + triangle[i][j];
                else if (j < triangle[i - 1].size()) {
                    dp[i][j] = min(dp[i - 1][j], dp[i - 1][j - 1]) + triangle[i][j];
                } else {
                    dp[i][j] = dp[i - 1][j - 1] + triangle[i][j];
                }
            }
        }

        for (int i = 0; i < n; i++) res = min(res, dp[n - 1][i]);

        return res;
    }
};
```
# 剑指 Offer 29. 顺时针打印矩阵
### 方法1
```
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.empty()) return {};
        int d = 1;
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};
        vector<int> ans;
        int n = matrix.size(), m = matrix[0].size();
        vector<vector<bool>> st(n, vector<bool>(m));
        int x = 0, y = 0, a = 0, b = 0;
        for (int i = 0; i < n * m; i++) {
            if (x >= 0 && x < n && y >= 0 && y < m && !st[x][y]) {
                ans.push_back(matrix[x][y]);
                st[x][y] = true;
                a = x, b = y;
                x = x + dx[d];
                y = y + dy[d];
            } else {
                d = (d + 1) % 4;
                x = a + dx[d];
                y = b + dy[d];
                st[x][y] = true;
                a = x, b = y;
                ans.push_back(matrix[x][y]);
                x += dx[d], y += dy[d];
            }
        }

        return ans;
    }
};
```
### 方法2
```
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        if (matrix.empty()) return {};
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};
        int row = matrix.size(), col = matrix[0].size();
        vector<vector<bool>> st(row, vector<bool>(col));
        vector<int> res;
        int d = 1, x = 0, y = 0;
        for (int i = 0; i < row * col; i++) {
            res.push_back(matrix[x][y]);
            st[x][y] = true;
            int a = x + dx[d], b = y + dy[d];
            if (a < 0 || a >= row || b < 0 || b >= col || st[a][b]) {
                d = (d + 1) % 4;
                a = x + dx[d];
                b = y + dy[d];
            }
            x = a, y = b;
        }

        return res;
    }
};
```
### 方法3
```
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        int dx[4] = {-1, 0, 1, 0};
        int dy[4] = {0, 1, 0, -1};
        int d = 1;
        int m = matrix.size(), n = matrix[0].size();
        vector<vector<bool>> st(m, vector<bool>(n));
        vector<int> res;
        int x = 0, y = 0;
        for (int i = 0; i < m * n; i++) {
            res.push_back(matrix[x][y]);
            st[x][y] = true;
            int a = x + dx[d], b = y + dy[d];
            if (a >= 0 && a < m && b >= 0 && b < n && !st[a][b]) {
                x = a, y = b;
            } else {
                d = (d + 1) % 4;
                x = x + dx[d];
                y = y + dy[d];
            }
        }

        return res;
    }
};
```
# 213. 打家劫舍 II
### 方法1：对头部元素分类讨论，两次遍历
```
class Solution {
public:
    int rob(vector<int>& nums) {
        int n = nums.size();
        if (n == 1) return nums[0];

        vector<int> dp(n), f(n);
        dp[0] = 0;
        dp[1] = nums[1];

        for (int i = 2; i < n; i++) {
            dp[i] = max(dp[i - 2] + nums[i], dp[i - 1]);
        }

        f[0] = nums[0];
        f[1] = nums[0];

        for (int i = 2; i < n - 1; i++) {
            f[i] = max(f[i - 2] + nums[i], f[i - 1]);
        }
        // cout << dp[n - 1] << " " << f[n - 2] << endl;
        return max(dp[n - 1], f[n - 2]);   
        // return f[n - 2];   
        // return dp[n - 1];   
    }
};
```
# 10. 正则表达式匹配
### dp
```
class Solution {
public:
    bool isMatch(string s, string p) {
        int n = s.size(), m = p.size();
        s = " " + s;
        p = " " + p;
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1));
        dp[0][0] = true;
        int i, j;
        for (i = 0; i <= n; i++) {
            for (j = 1; j <= m; j++) {
                if (p[j] == '*') {
                    dp[i][j] = dp[i][j - 1] || dp[i][j - 2] || (i && dp[i - 1][j - 1] && (p[j - 1] == '.' || p[j - 1] == s[i]));
                } else {
                    dp[i][j] = i && dp[i - 1][j - 1] &&  (((s[i] == p[j]) || p[j] == '.'));
                }
                // cout << i << " " << j << " " << dp[i][j] << endl;
            }
            
        }

        return dp[n][m];
    }
};
```
# 349. 两个数组的交集
### 方法1 set
```
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        vector<int> ans;
        unordered_set<int> s1(nums1.begin(), nums1.end());
        unordered_set<int> s2(nums2.begin(), nums2.end());
        for (auto x : s1) {
            if (s2.count(x)) {
                ans.push_back(x);
            }
        }

        return ans;
    }
};
```
### 方法2：排序 + 双指针
```
class Solution {
public:
    vector<int> intersection(vector<int>& nums1, vector<int>& nums2) {
        int l = 0, r = 0;
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());
        vector<int> ans;
        while (l < nums1.size() && r < nums2.size()) {
            if (nums1[l] == nums2[r]) {
                if (ans.empty() || ans.back() != nums2[r]) {
                    ans.push_back(nums2[r]);
                }
                l++, r++;
            } else if (nums1[l] > nums2[r]) {
                r++;
            } else {
                l++;
            }
        }

        return ans;
    }
};
```
# 106. 从中序与后序遍历序列构造二叉树
### 方法1
```
class Solution {
public:
    unordered_map<int, int> h;
    vector<int> in, post;
    int n;
    TreeNode* buildTree(vector<int>& inorder, vector<int>& postorder) {
        for (int i = 0; i < inorder.size(); i++) {
            h[inorder[i]] = i;
        }
        in = inorder, post = postorder;
        n = inorder.size() - 1;
        int l = 0, r = inorder.size() - 1;
        return build(0, n, 0, n);
    }
    TreeNode* build(int il, int ir, int pl, int pr)
    {
        if (il > ir) return nullptr;
        int val = post[pr];
        auto root = new TreeNode(val);
        int id = h[val];
        root->left = build(il, id - 1, pl, pl + id - 1 - il);
        root->right = build(id + 1, ir, pl + id - 1 - il + 1, pr - 1);

        return root;
    }
};
```
# 400. 第N个数字
##数位计算
```
class Solution {
public:
    int findNthDigit(int n) {
        typedef long long LL;
        LL k = 1, s = 1, t = 9;
        while (n > k * t) {
            n -= k * t;
            k++;
            s *= 10;
            t *= 10;
        }

        s += (n + k - 1) / k - 1;
        n = n % k ? n % k : k;

        return to_string(s)[n - 1] - '0';
    }
};
```
# 面试题61. 扑克牌中的顺子
### 方法1
```
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        int z = 0;
        for (auto x : nums) {
            if (x == 0) z++;
        }

        sort(nums.begin(), nums.end());
        for (int i = z + 1; i < nums.size(); i++) {
            if (nums[i] - nums[i - 1] == 0) return false;
            else if (nums[i] - nums[i - 1] == 1) continue;
            else {
                if (nums[i] - nums[i - 1] - 1 <= z) {
                    z -= (nums[i] - nums[i - 1] - 1);
                    continue;
                } else {
                    return false;
                }
            }
        }

        return true;
    }
};
```
### 方法2
```
class Solution {
public:
    bool isStraight(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] && nums[i] == nums[i - 1]) return false;
        }

        for (auto x : nums) {
            if (x) {
                return nums.back() - x <= 4;
            }
        }

        return true;
    }
};
```
# 678. 有效的括号字符串
### codetop方法1：括号判断的条件
```
class Solution {
public:
    bool checkValidString(string s) {
        int l = 0, r = 0;
        for (auto x : s) {
            if (x == '(') {
                l++, r++;
            } else if (x == ')') {
                l--, r--;
            } else if (x == '*') {
                l--, r++;
            }
            l = max(0, l);
            if (l > r) return false;
        }

        return l == 0;
    }
};
```
# 347. 前 K 个高频元素
### 哈希表 + 计数
```
class Solution {
public:
    vector<int> topKFrequent(vector<int>& nums, int k) {
        unordered_map<int, int> h;
        int n = 0;
        for (auto x : nums) h[x]++, n = max(n, h[x]);
        
        vector<vector<int>> g(n + 1);

        for (auto [k, v]: h) {
            // cout << k << " " << v << endl;
            g[v].push_back(k);
        }
        int t = 0;
        vector<int> ans;

        for (int i = n; i >= 1; i--) {
            for (auto x : g[i]) {
                ans.push_back(x);
                t++;
                if (t >= k) return ans;
            }
        }

        return ans;
    }
};
```

# 611. 有效三角形的个数
### 方法1：二分 
```
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int cnt = 0, n = nums.size();
        for (int i = 0; i < n; i++) {
            for (int j = i + 1; j < n; j++) {
                int l = j + 1, r = n - 1, k = j;
                int v = nums[i] + nums[j];
                while (l < r) {
                    int mid = (l + r) >> 1;
                    if (v > nums[mid]) l = mid + 1, k = mid;
                    else r = mid;
                }
                if (v > nums[r])
                    cnt += max(r - j, 0);
                else 
                    cnt += max(r - j - 1, 0);
                // cout << i << " " << j << " " << r << endl;
            }  
        }

        return cnt;
    }
};
```
### 方法2 双指针
```
class Solution {
public:
    int triangleNumber(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        int cnt = 0, n = nums.size();
        for (int i = 0; i < n; i++) {
            int k = i;
            for (int j = i + 1; j < n; j++) {
                while (k + 1 < n && nums[i] + nums[j] > nums[k + 1]) k++;
                // cout << k << " "
                cnt += max(k - j, 0);
            }  
        }

        return cnt;
    }


};
```
# 补充题9. 36进制加法
```
#include <iostream>
#include <algorithm>
using namespace std;

char getChar(int n)
{
    if (n <= 9)
        return n + '0';
    else
        return n - 10 + 'a';
}
int getInt(char ch)
{
    if ('0' <= ch && ch <= '9')
        return ch - '0';
    else
        return ch - 'a' + 10;
}
string add36Strings(string num1, string num2)
{
    int carry = 0;
    int i = num1.size() - 1, j = num2.size() - 1;
    int x, y;
    string res;
    while (i >= 0 || j >= 0 || carry)
    {
        x = i >= 0 ? getInt(num1[i]) : 0;
        y = j >= 0 ? getInt(num2[j]) : 0;
        int temp = x + y + carry;
        res += getChar(temp % 36);
        carry = temp / 36;
        i--, j--;
    }
    reverse(res.begin(), res.end());
    return res;
}

int main()
{
    string a = "1b", b = "2x", c;
    c = add36Strings(a, b);
    cout << c << endl;
}
```
# 剑指 Offer 52. 两个链表的第一个公共节点
```
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *a, ListNode *b) {
        auto p1 = a, p2 = b;
        while (p1 != p2) {
            p1 = (p1 == NULL ? b : p1->next);
            p2 = (p2 == NULL ? a : p2->next);
        }

        return p1;
    }
};
```

# 面试题 02.05. 链表求和
```
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        auto dummy = new ListNode();
        auto p = dummy;
        int c = 0;
        while (l1 || l2 || c) {
            int t = 0;
            if (l1) t += l1->val, l1 = l1->next;
            if (l2) t += l2->val, l2 = l2->next;
            t += c;
            auto node = new ListNode(t % 10);
            p->next = node;
            p = p->next;
            c = t / 10;
        }

        return dummy->next;
    }
};
```
# 887. 鸡蛋掉落
### dp
```
int f[10010][105];
class Solution {
public:
    int superEggDrop(int k, int n) {
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= k; j++) {
                f[i][j] = f[i - 1][j - 1] + f[i - 1][j] + 1;
            }
            if (f[i][k] >= n) return i;
        }

        return 0;
    }
};
```
# 剑指 Offer 39. 数组中出现次数超过一半的数字
### 投票算法
```
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int ans = nums[0], cnt = 1;
        for (int i = 1; i < nums.size(); i++) {
            if (nums[i] == ans) {
                cnt++;
            } else {
                cnt--;
                if (!cnt) {
                    ans = nums[i];
                    cnt++;
                }
            }
        }

        return ans;
    }
};
```
# 168. Excel表列名称
```
class Solution {
public:
    string convertToTitle(int n) {
        string s;
        do {
            n--;
            int v = n % 26;
            s += ('A' + v);
            n /= 26;
        } while (n);

        reverse(s.begin(), s.end());

        return s;
    }
};
```
# 673. 最长递增子序列的个数
```
//2022.12.22 周四
class Solution {
public:
    int findNumberOfLIS(vector<int>& nums) {
        int n = nums.size();
        vector<int> dp(n, 1), cnt(n, 1);
        int ans = 1;
        for (int i = 1; i < n; i++) {
            for (int j = 0; j < i; j++) {
                if (nums[i] > nums[j]) {
                    // dp[i] = max(dp[j] + 1, dp[i]);
                    if (dp[j] + 1 > dp[i]) {
                        cnt[i] = cnt[j];
                        dp[i] = dp[j] + 1;
                    } else if (dp[j] + 1 == dp[i]) {
                        cnt[i] += cnt[j];
                    }
                }
            }
            ans = max(ans, dp[i]);
        }
        // cout << "ans: " << ans << endl;
        int res = 0;
        for (int i = 0; i < n; i++) {
            if (dp[i] == ans) {
                res += cnt[i];
            }
        }


        return res;
    }
};
```

# 674. 最长连续递增序列
```
//2022.12.22 周四
class Solution {
public:
    int findLengthOfLCIS(vector<int>& nums) {
        int n = nums.size(), ans = 1, i = 0;
        while (i < n) {
            int j = i + 1;
            while (j < n && nums[j] > nums[j - 1]) j++;
            ans = max(ans, j - i);
            i = j;
        }

        return ans;
    }
};
```

# 1047. 删除字符串中的所有相邻重复项
### 栈的妙用
```
class Solution {
public:
    string removeDuplicates(string s) {
        stack<char> st;
        for (auto x : s) {
            if (st.empty() || st.top() != x) {
                st.push(x);
            } else {
                st.pop();
            }
        }
        string res;
        while (st.size()) {
            res += st.top();
            st.pop();
        }
        reverse(res.begin(), res.end());

        return res;
    }
};
```
# 	
111. 二叉树的最小深度
### bfs
```
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;
        queue<TreeNode*> q;
        int ans = 1;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                if (node->left == nullptr && node->right == nullptr) {
                    return ans;
                }
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            ans++;
        }

        return ans;
    }
};
```
### DFS
```
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;
        int l = minDepth(root->left);
        int r = minDepth(root->right);
        if (l && r) return min(l, r) + 1;
        if (l && !r) return l + 1;
        if (!l && r) return r + 1;
        return 1;
    }
};
```
### DFS
```
class Solution {
public:
    int minDepth(TreeNode* root) {
        if (!root) return 0;
        if (!root->left) return minDepth(root->right) + 1;
        if (!root->right) return minDepth(root->left) + 1;

        return min(minDepth(root->left), minDepth(root->right)) + 1;
    }
};
```
# 442. 数组中重复的数据
### 方法1：哈希表
```
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> res;
        unordered_map<int, int> h;
        for (auto x : nums) h[x]++;
        for (auto [k, v] : h) {
            if (v == 2) res.push_back(k);
        }

        return res;
    }
};
```

### 方法2：原地修改数组
```
class Solution {
public:
    vector<int> findDuplicates(vector<int>& nums) {
        vector<int> ans;
        for (auto x : nums) {
            x = abs(x);
            if (nums[x - 1] > 0) {
                nums[x - 1] = nums[x - 1] * -1;
            } else {
                ans.push_back(x);
            }
        }
 

        return ans;
    }
};
```

# 448. 找到所有数组中消失的数字
### 方法1
```
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        int n = nums.size();
        vector<bool> flag(n + 1, false);
        for (auto& x: nums) {
            flag[x] = true;
        }
        vector<int> res;
        for (int i = 1; i <= n; i++) {
            if (flag[i] == false) {
                res.push_back(i);
            }
        }

        return res;
    }
};
```
### 方法2
```
class Solution {
public:
    vector<int> findDisappearedNumbers(vector<int>& nums) {
        for (auto x : nums) {
            x = abs(x);
            if (nums[x - 1] > 0) nums[x - 1] *= -1;
        }

        vector<int> ans;
        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] > 0) ans.push_back(i + 1);
        }

        return ans;
    }
};
```

# 71. 简化路径
### 栈的运用
```
class Solution {
public:
    string simplifyPath(string path) {
        string res, name;
        if (path.back() != '/') path += '/';

        for (auto c : path) {
            if (c != '/') name += c;
            else {
                if (name == "..") {
                    while (res.size() && res.back() != '/') res.pop_back();
                    if (res.size()) res.pop_back();
                } else if (name != "" && name != ".") {
                    res += "/" + name;
                }
                name.clear();
            }
        }
        if (res.empty()) res = "/";

        return res;
    }
};
```
# 134. 加油站
### 方法1
```
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        vector<int> sum;
        for (int i = 0; i < 2 * n; i++) {
            sum.push_back(gas[i % n] - cost[i % n]);
        }

        int start = 0, end = 0, tot = 0;
        while (start < n && end < 2 * n) {
            tot += sum[end];
            while (tot < 0) {
                tot -= sum[start];
                start++;
            }

            if (end - start + 1 == n)
                return start;
            end++;
        }

        return -1;
    }
};
```

### 方法2
```
class Solution {
public:
    int canCompleteCircuit(vector<int>& gas, vector<int>& cost) {
        int n = gas.size();
        for (int i = 0, j; i < n; ) {  // 枚举起点
            int left = 0;
            for (j = 0; j < n; j ++ ) {
                int k = (i + j) % n;
                left += (gas[k] - cost[k]);
                if (left < 0) break;
            }
            if (j == n) return i;
            i = i + j + 1;
        }

        return -1;
    }
};
```
# 86. 分隔链表
### 双指针
```
class Solution {
public:
    ListNode* partition(ListNode* head, int x) {
        auto p0 = new ListNode(), p1 = new ListNode();
        auto h0 = p0, h1 = p1;
        while (head) {
            if (head->val < x) {
                h0->next = head;
                h0 = h0->next;
            } else {
                h1->next = head;
                h1 = h1->next;
            }
            head = head->next;
        }

        h0->next = p1->next;
        h1->next = nullptr;

        return p0->next;
    }
};
```
# 1004. 最大连续1的个数 III
### 双指针
```
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int res = 0, cnt = 0;
        for (int i = 0, j = 0; i < nums.size(); i++) {
            if (nums[i] == 0) cnt++;
            while (cnt > k) {
                if (nums[j] == 0) cnt--;
                j++;
            }

            res = max(res, i - j + 1);
        }

        return res;
    }
};
```
# 剑指 Offer 34. 二叉树中和为某一值的路径
### dfs
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    void dfs(TreeNode* root, int target, int s)
    {
        if (!root) return;
        if (root->left == nullptr && root->right == nullptr && s + root->val == target) {
            path.push_back(root->val);
            ans.push_back(path);
            path.pop_back();
            return;
        }
        path.push_back(root->val);
        dfs(root->left, target, s + root->val);
        dfs(root->right, target, s + root->val);
        path.pop_back();
    }
    vector<vector<int>> pathSum(TreeNode* root, int target) {
        dfs(root, target, 0);

        return ans;
    }
};
```
# 剑指 Offer 53 - I. 在排序数组中查找数字 I
### 方法1：哈希表
```
class Solution {
public:
    int search(vector<int>& nums, int target) {
        unordered_map<int, int> h;
        for (auto x : nums) h[x]++;

        return h[target];
    }
};
```
### 方法2：库函数
```
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int a = lower_bound(nums.begin(), nums.end(), target) - nums.begin();
        int b = upper_bound(nums.begin(), nums.end(), target) - nums.begin();
        return b - a;
    }
};
```
### 方法3：二分
```
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if (nums.size() == 0) return 0;
        int a = 0, b = 0;
        int l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] >= target) r = mid;
            else l = mid + 1;
        }
        a = l;

        l = 0, r = nums.size() - 1;
        while (l < r) {
            int mid = l + r >> 1;
            if (nums[mid] > target) r = mid;
            else l = mid + 1;
        }
        if (nums[l] == target) b = l + 1;
        else b = l;

        return b - a;
    }
};
```
# 279. 完全平方数
### dp
```
// 2022.12.27 周二
class Solution {
public:
    int numSquares(int n) {
        vector<int> dp(n + 1, INT_MAX);
        dp[0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= i / j; j++) {
                dp[i] = min(dp[i], dp[i - j * j] + 1);
            }
        }

        return dp[n];
    }
};
```
# 63. 不同路径 II
### dp
```
// 2022.12.27 周二
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& g) {
        int n = g.size(), m = g[0].size();
        vector<vector<int>> dp(n, vector<int>(m));
        for (int i = 0; i < m; i++) {
            if (g[0][i] == 0) dp[0][i] = 1;
            else break;
        }

        for (int i = 0; i < n; i++) {
            if (g[i][0] == 0) dp[i][0] = 1;
            else break;
        }
        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                if (g[i][j] == 0) dp[i][j] = dp[i - 1][j] + dp[i][j - 1];
                else dp[i][j] = 0;
            }
        }

        return dp[n - 1][m - 1];
    }
};
```
# 44. 通配符匹配
### dp
```
class Solution {
public:
    bool isMatch(string s, string p) {
        int n = s.size(), m = p.size();
        s = " " + s;
        p = " " + p;
        vector<vector<bool>> dp(n + 1, vector<bool>(m + 1));
        dp[0][0] = true;
        for (int i = 0; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (p[j] == s[i] || p[j] == '?') {
                // if (p[j] != '*') {
                    // dp[i][j] = i && dp[i - 1][j - 1] && (p[j] == s[i] || p[j] == '?');
                    dp[i][j] = i && dp[i - 1][j - 1];
                } else if (p[j] == '*') {
                    dp[i][j] = dp[i][j - 1] || (i && dp[i - 1][j]);
                }
            }
        }

        return dp[n][m];
    }
};
```
# 679. 24 点游戏
### dfs
```
// 2022.12.28 周三
class Solution {
public:
    bool judgePoint24(vector<int>& cards) {
        vector<double> a(cards.begin(), cards.end());
        return dfs(a);
    }

    bool dfs(vector<double> nums)
    {
        if (nums.size() == 1) {
            return fabs(nums[0] - 24) < 1e-8;
        }

        for (int i = 0; i < nums.size(); i++) {
            for (int j = 0; j < nums.size(); j++) {
                if (i != j) {
                    double a = nums[i], b = nums[j];
                    if (dfs(get(nums, i, j, a + b))) return true;
                    if (dfs(get(nums, i, j, a - b))) return true;
                    if (dfs(get(nums, i, j, a * b))) return true;
                    if (b && dfs(get(nums, i, j, a / b))) return true;
                }
            }
        }

        return false;
    }

    vector<double> get(vector<double> &nums, int i, int j, double x) {
        vector<double> res;
        for (int k = 0; k < nums.size(); k++) {
            if (k != i && k != j) res.push_back(nums[k]);
        }

        res.push_back(x);

        return res;
    }
};
```
# 37. 解数独
### dfs
```
class Solution {
public:
    bool row[9][10], col[9][10], cell[3][3][10];
    void solveSudoku(vector<vector<char>>& board) {
        memset(row, 0, sizeof row);
        memset(col, 0, sizeof col);
        memset(cell, 0, sizeof cell);

        for (int i = 0; i < 9; i++) {
            for (int j = 0; j < 9; j++) {
                if (board[i][j] != '.') {
                    int u = board[i][j] - '0';
                    row[i][u] = col[j][u] = cell[i / 3][j / 3][u] = true;
                }
            }
        }
        dfs(board, 0, 0);
    }

    bool dfs(vector<vector<char>>& g, int x, int y) {
        if (y == 9) {
            y = 0;
            x++;
            if (x == 9) return true;
        }

        if (g[x][y] != '.') return dfs(g, x, y + 1);

        for (int i = 1; i <= 9; i++) {
            if (!row[x][i] && !col[y][i] && !cell[x / 3][y / 3][i]) {
                row[x][i] = col[y][i] = cell[x / 3][y / 3][i] = true;
                g[x][y] = '0' + i;
                if (dfs(g, x, y + 1)) return true;
                g[x][y] = '.';
                row[x][i] = col[y][i] = cell[x / 3][y / 3][i] = false;
            }
        }

        return false;
    }
};
```
# 556. 下一个更大元素 III
```
class Solution {
public:
int nextGreaterElement(int n) {
        string s = to_string(n);
        int len = s.size();
        int k = len - 1;
        while (k && s[k - 1] >= s[k]) k--;
        if (k == 0) return -1;
        int j = k - 1;
        while (j + 1 < len && s[j + 1] > s[k - 1]) j++;
        swap(s[k - 1], s[j]);
        reverse(s.begin() + k, s.end());
        long long num = stol(s);
        if (num > INT_MAX) return -1;

        return num;
    }
};
```
# 459. 重复的子字符串
### 方法1
```
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        int n = s.size();
        string t;
        for (int i = 0; i < n; i++) {
            if (t.size() && t[0] == s[i]) {
                int len = t.size();
                if (n % len == 0) {
                    int m = n / len;
                    string temp;
                    for (int i = 0; i < m; i++) {
                        temp += t;
                    }
                    if (temp == s) return true;
                }
            }
            t += s[i];
        }

        return false;
    }
};
```
### 方法2
```
class Solution {
public:
    bool repeatedSubstringPattern(string s) {
        string t = s;
         s += s;
         int n = s.size();
         s = s.substr(1, n - 2);
        //  cout << s << endl;
        auto x = s.find(t);
        if (x != s.npos) return true;

        return false;
    }
};
```

# 27. 移除元素
### 方法1
```
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int l = 0, r = 0;
        while (l < nums.size()) {
            if (nums[l] != val) {
                nums[r++] = nums[l];
            }
            l++;
        }

        return r;
    }
};
```
# 844. 比较含退格的字符串
### codetop方法1
```
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        stack<char> st;
        string s2, t2;
        for (auto x : s) {
            if (st.empty()) {
                if (x != '#')
                    st.push(x);
            } else if (x == '#') st.pop();
            else st.push(x);
        }
        while (st.size()) {
            s2 += st.top();
            st.pop();
        }

        for (auto x : t) {
            if (st.empty()) {
                if (x != '#')
                    st.push(x);
            } else if (x == '#') st.pop();
            else st.push(x);
        }
        while (st.size()) {
            t2 += st.top();
            st.pop();
        }
        // cout << s2 << " " << t2 << endl;
        return s2 == t2;
    }
};
```
### codetop 方法2
```
class Solution {
public:
    bool backspaceCompare(string s, string t) {
        string s2, t2;
        for (auto x : s) {
            if (x != '#') s2 += x;
            else {
                if (s2.size()) s2.pop_back();
            }
        }

        for (auto x : t) {
            if (x != '#') t2 += x;
            else {
                if (t2.size()) t2.pop_back();
            }
        }

        return s2 == t2;
    }
};
```
# 977. 有序数组的平方
### 方法1：暴力算法
```
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        vector<int> ans;
        for (auto x : nums) {
            ans.push_back(x * x);
        }
        sort(ans.begin(), ans.end());
        return ans;
    }
};
```
### 方法2 注意最大的数只可能在两端，所以可以使用双指针
```
class Solution {
public:
    vector<int> sortedSquares(vector<int>& nums) {
        int n = nums.size();
        vector<int> ans(n);
        int k = n - 1;
        int l = 0, r = n - 1;
        while (l <= r) {
            if (nums[l] * nums[l] >= nums[r] * nums[r]) {
                ans[k--] = nums[l] * nums[l];
                l++;
            } else {
                ans[k--] = nums[r] * nums[r];
                r--;
            }
        }

        return ans;
    }
};
```
# 904. 水果成篮
### 滑动窗口 + 哈希
```
class Solution {
public:
    int totalFruit(vector<int>& s) {
        int n = s.size();
        int ans = 0;
        unordered_map<int, int> h;
        for (int i = 0, j = 0; i < n; i++) {
            if (h.size() == 0) {
                h[s[i]]++;
                ans = max(ans, i - j + 1);
                // cout << "/:" << i << " " << j << " " << ans << endl;
            } else if (h.size() == 1) {
                if (h.count(s[i]) == 0) {
                    h[s[i]]++;
                } else {
                    h[s[i]]++;
                }
                ans = max(ans, i - j + 1);
                // cout << "//:" << i << " " << j << " " << ans << endl;
            } else if (h.size() == 2) {
                if (h.count(s[i])) {
                    h[s[i]]++;
                    ans = max(ans, i - j + 1);
                    // cout << "///0:" << i << " " << j << " " << ans << endl;
                    // cout << "///0: h[s[i]] = " << h[s[i]] << endl;
                } else {
                    while (h[s[j]] > 0) {
                        h[s[j]]--;
                        // cout << "*** i:" << i << ", j:" << j << ", h[s[j]] = " << h[s[j]] << endl;
                        if (h[s[j]]) j++;
                        else {
                            h.erase(s[j++]);
                            break;
                        }
                    }
                    
                    h[s[i]]++;
                    ans = max(ans, i - j + 1);
                    // cout << "///1:" << i << " " << j << " " << ans << endl;
                }  
            }
        }

        return ans;
    }
};
```
### 对上述代码的优化
```
class Solution {
public:
    int totalFruit(vector<int>& s) {
        int n = s.size();
        int ans = 0;
        unordered_map<int, int> h;
        for (int i = 0, j = 0; i < n; i++) {
            if (h.size() == 0 || h.size() == 1) {
                h[s[i]]++;
                ans = max(ans, i - j + 1);
            } else if (h.size() == 2) {
                if (h.count(s[i])) {
                    h[s[i]]++;
                    ans = max(ans, i - j + 1);
                } else {
                    while (h[s[j]] > 0) {
                        h[s[j]]--;
                        if (h[s[j]]) j++;
                        else {
                            h.erase(s[j++]);
                            break;
                        }
                    }
                    h[s[i]]++;
                    ans = max(ans, i - j + 1);
                }  
            }
        }
        return ans;
    }
};
```
### 方法2
```
class Solution {
public:
    int totalFruit(vector<int>& fruits) {
        unordered_map<int, int> h;
        int ans = 0;
        for (int i = 0, j = 0, s = 0; i < fruits.size(); i++) {
            if (++h[fruits[i]] == 1) s++;

            while (s > 2) {
                if (--h[fruits[j]] == 0) s--;
                j++;
            }
            ans = max(ans ,i - j + 1);
        }

        return ans;
    }
};
```
# 203. 移除链表元素
### 方法1
```
class Solution {
public:
    ListNode* removeElements(ListNode* head, int val) {
        auto dummy = new ListNode();
        auto p = dummy;
        while (head) {
            if (head->val != val) {
                p->next = head;
                p = p->next;
            }

            head = head->next;
        }
        p->next = nullptr;
        return dummy->next;
    }
};
```
# 242. 有效的字母异位词
### 方法1：排序
```
class Solution {
public:
    bool isAnagram(string s, string t) {
        sort(s.begin(), s.end());
        sort(t.begin(), t.end());

        return s == t;
    }
};
```
### 方法2：哈希表
```
class Solution {
public:
    bool isAnagram(string s, string t) {
        if (s.size() != t.size()) return false;
        unordered_map<char, int> hs, ht;
        for (auto x : s) {
            hs[x]++;
        }

        for (auto x : t) {
            ht[x]++;
        }

        for (auto[k, v] : hs) {
            if (ht[k] != v) return false;
        }

        return true;
    }
};
```
# 383. 赎金信
### 方法1：哈希表
```
2022.1.8 周日
class Solution {
public:
    bool canConstruct(string s, string t) {
        unordered_map<char, int> hs, ht;
        for (auto x : s) hs[x]++;
        for (auto x : t) ht[x]++;
        for (auto [k, v]: hs) {
            if (v <= ht[k]) continue;
            else return false;
        }

        return true;
    }
};
```
# 49.字母异位词分组
###  方法1
```
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        vector<vector<string>> ans;
        unordered_map<string, vector<string>> h;
        for (auto x : strs) {
            string t = x;
            sort(t.begin(), t.end());
            h[t].push_back(x);
        }

        for (auto [k, v]: h) {
            ans.push_back(v);
        }

        return ans;
    }
};
```
### 方法2
```
class Solution {
public:
    vector<vector<string>> groupAnagrams(vector<string>& strs) {
        unordered_map<string, int> hash;
        int id = 0;
        for (auto x: strs) {
            sort(x.begin(), x.end());
            if (hash.count(x) == 0) {
                hash[x] = id++;
            }
        }
        vector<vector<string>> res(id, vector<string>());
        for (auto x: strs) {
            auto t = x;
            sort(t.begin(), t.end());
            int idx = hash[t];
            res[idx].push_back(x);
        }

        return res;
    }
};
```
# 438.找到字符串中所有字母异位词
### 哈希表 + 滑动窗口
```
class Solution {
public:
    vector<int> findAnagrams(string s, string p) {
        vector<int> ans;
        int ns = s.size(), np = p.size();
        unordered_map<char, int> hp;
        int cnt = 0;
        
        for (auto x : p) hp[x]++;
        int n = hp.size();
        for (int i = 0, j = 0; i < ns; i++) {
            if (--hp[s[i]] == 0) cnt++;
            if (i - j + 1 > np) {
                if (hp[s[j]] == 0) cnt--;
                hp[s[j++]]++;
            }
            if (cnt == n) ans.push_back(j);
        }

        return ans;
    }
};
```
# 350. 两个数组的交集 II
### 方法1：哈希表
```
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        vector<int> ans;
        unordered_map<int, int> h1, h2;
        for (auto x : nums1) h1[x]++;
        for (auto x : nums2) h2[x]++;
        for (auto [k, v] : h1) {
            // cout << "k: " << k << endl;
            if (h2.count(k)) {
                int v2 = h2[k];
                int t = min(v, v2);
                for (int i = 0; i < t; i++) {
                    ans.push_back(k);
                }
            } 
        }

        return ans;
    }
};
```
### 方法2：双指针
```
class Solution {
public:
    vector<int> intersect(vector<int>& nums1, vector<int>& nums2) {
        int l = 0, r = 0;
        vector<int> ans;
        sort(nums1.begin(), nums1.end());
        sort(nums2.begin(), nums2.end());

        while (l < nums1.size() && r < nums2.size()) {
            if (nums1[l] == nums2[r]) {
                ans.push_back(nums1[l]);     
                l++;
                r++;
            } else if (nums1[l] > nums2[r]) {
                r++;
            } else {
                l++;
            }
        }

        return ans;
    }
};
```
# 202. 快乐数
# 方法1
```
class Solution {
public:
    bool isHappy(int n) {
        if (n == 1) return true;
        unordered_set<int> h;
        h.insert(n);

        while (n != 1) {
            int t = 0;
            string s = to_string(n);
            for (auto x : s) {
                int v = x - '0';
                t += v * v;
            }
            if (t == 1) return true;

            if (h.count(t)) {
                return false;
            } else {
                h.insert(t);
            }
            n = t;
        }

        return false;
    }
};
```

### 方法2：快慢指针
```
class Solution {
public:
    int get(int n) {
        string s = to_string(n);
        int t = 0;
        for (auto x : s) {
            int v = x - '0';
            t += v * v;
        }

        return t;
    }


    bool isHappy(int n) {
        int f = get(get(n)), s = get(n);
        while (f != s) {
            f = get(get(f)), s = get(s);
        }

        return f == 1;
    }
};
```
# 707. 设计链表
### 方法1：注意设计一个函数，都在某个节点之前插入，还有设置虚拟头尾节点
```
class MyLinkedList {
public:
    struct DoubleList {
        int val;
        DoubleList* next;
        DoubleList* prev;
        int index;
        DoubleList(int _val, int _index): val(_val), index(_index), 
                                          prev(nullptr), next(nullptr) 
        {

        } 
    };
    DoubleList *head, *tail;
    void printDebug() {
        auto t = head->next;
        while (t != tail) {
            printf("val:%d, index:%d\n", t->val, t->index);
            t = t->next;
        }
    }
    MyLinkedList() {
        head = new DoubleList(0, -1);
        tail = new DoubleList(0, 10000);
        head->next = tail;
        tail->prev = head;
    }

    // 在curNode之前添加节点
    void addNode(DoubleList* curNode, int val, int index) {
        DoubleList* node = new DoubleList(val, index);
        auto t = curNode->prev;
        node->next = curNode;
        curNode->prev = node;
        node->prev = t;
        t->next = node;
        auto p = node;
        int id = index;
        while (p != tail) {
            p->index = id++;
            p = p->next;
        }
    }
    
    int get(int index) {
        int len = tail->prev->index;
        if (index > len) return -1;

        auto p = head->next;
        while (p != tail) {
            if (p->index != index) {
                p = p->next;
            } else {
                return p->val;
            }
        }

        return -1;
    }
    
    void addAtHead(int val) {
        addNode(head->next, val, 0);
        // printf("addAtHead\n");
        // printDebug();
    }
    
    void addAtTail(int val) {
        addNode(tail, val, tail->prev->index + 1);
        // printf("addAtTail\n");
        // printDebug();
    }
    
    void addAtIndex(int index, int val) {
        int len = tail->prev->index + 1;
        if (len == index) {
            addNode(tail, val, tail->prev->index + 1);
        } else {
            auto ph = head->next, pt = tail;
            while (ph != tail) {
                if (ph->index != index) {
                    ph = ph->next;
                } else {
                    addNode(ph, val, index);
                }
            }
        }

        // printf("addAtIndex\n");
        // printDebug();
    }
    
    void deleteAtIndex(int index) {
        int len = tail->prev->index;
        if (index > len) return;
        auto p = head->next;
        while (p->index != index) {
            p = p->next;
        }
        auto t = p->prev;
        p->prev->next = p->next;
        p->next->prev = p->prev;
        int id = t->index;
        while (t != tail) {
            t->index = id++;
            t = t->next;
        }

        // printf("deleteAtIndex\n");
        // printDebug();
    }
};
```

# 18. 四数之和
### 方法1：双指针
```
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        int len = nums.size();
        vector<vector<int>> ans;
        sort(nums.begin(), nums.end());
        for (int i = 0; i < len - 3; i++) {
            for (int j = i + 1; j < len - 2; j++) {
                long long sum = nums[i] + nums[j];
                int l = j + 1, r = len - 1;
                while (l < r) {
                    vector<int> temp;
                    if ((sum + nums[l] + nums[r]) == target) {
                        temp.push_back(nums[i]);
                        temp.push_back(nums[j]);
                        temp.push_back(nums[l]);
                        temp.push_back(nums[r]);
                        ans.push_back(temp);
                        l++;
                        r--;
                    } else if (sum + nums[l] + nums[r] > target) {
                        r--;
                    } else {
                        l++;
                    }
                }
            }
        }
        sort(ans.begin(), ans.end());
        ans.erase(unique(ans.begin(), ans.end()), ans.end());
        return ans;
    }
};
```
### 方法2：双指针
```
class Solution {
public:
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        vector<vector<int>> ans;
        int n = nums.size();
        sort(nums.begin(), nums.end());
        for (int i = 0; i < n; i++) {
            if (i && nums[i] == nums[i - 1]) continue;
            for (int j = i + 1; j < n; j++) {
                if (j != i + 1 && nums[j] == nums[j - 1]) continue;
                int l = j + 1, r = n - 1;
                while (l < r) {
                    long long v = (long long)nums[i] + nums[j] + nums[l] + nums[r];
                    // printf("i = %d, j = %d, l = %d, r = %d, v = %d\n", i, j, l, r, v);
                    if (v == target) {
                        vector<int> temp({nums[i], nums[j], nums[l], nums[r]});
                        ans.push_back(temp);
                        while (l + 1 < r && nums[l + 1] == nums[l]) l++;
                        while (r - 1 > l && nums[r - 1] == nums[r]) r--;
                        l++, r--;
                    } else if (v > target) r--;
                    else l++;
                }
            }
        }

        return ans;
    }
};
```
# 454. 四数相加 II
### 方法1：哈希表
```
class Solution {
public:
    int fourSumCount(vector<int>& nums1, vector<int>& nums2, vector<int>& nums3, vector<int>& nums4) {
        int n = nums1.size();
        unordered_map<int, int> h;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int v = nums1[i] + nums2[j];
                h[v]++;
            }
        }
        int ans = 0;
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                int v = nums3[i] + nums4[j];
                int s = 0 - v;
                if (h.count(s)) {
                    ans += h[s];
                }
            }
        }

        return ans;
    }
};
```
# 344. 反转字符串
### 双指针
```
class Solution {
public:
    void reverseString(vector<char>& s) {
        int l = 0, r = s.size() - 1;
        while (l < r) {
            swap(s[l], s[r]);
            l++;
            r--;
        }
    }
};
```
# 541. 反转字符串 II
### 方法1
```
class Solution {
public:
    string reverseStr(string s, int k) {
        int n = s.size(), i = 0;
        string ans;
        for (int i = 0; i < n; ) {
            if (i + 2 * k < n) {
                for (int j = i + k; j > i;) {
                    ans += s[--j];
                } 

                for (int j = i + k; j < i + 2 * k; j++) {
                    ans += s[j];
                }
                i += 2 * k;
            } else if (i + k <= n) {
                for (int j = i + k; j > i;) {
                    ans += s[--j];
                } 

                i += k;
                for (int j = i; j < n; j++) {
                    ans += s[j];
                }

                i = n;
            } else {
                for (int j = n - 1; j >= i; j--) {
                    ans += s[j];
                }

                i = n;
            }
        }

        return ans;
    }
};
```
### 方法2
```
class Solution {
public:
    string reverseStr(string s, int k) {
        for (int i = 0; i < s.size(); i += 2 * k) {
            if (i + k <= s.size()) {
                reverse(s.begin() + i, s.begin() + i + k);
            } else {
                reverse(s.begin() + i, s.end());
            }
        }

        return s;
    }
};
```

# 剑指 Offer 05. 替换空格
### 方法1 for循环
```
class Solution {
public:
    string replaceSpace(string s) {
        string res;
        for (auto x : s) {
            if (x == ' ') res += "%20";
            else res += x;
        }

        return res;
    }
};
```
### 方法2 双指针
```
class Solution {
public:
    string replaceSpace(string s) {
        int cntSpace = 0;
        for (auto x : s) {
            if (x == ' ') cntSpace++;
        }
        int n = s.size();
        int len = s.size() + cntSpace * 2;
        s.resize(len);
        for (int i = len - 1, j = n - 1; j >= 0; j--) {
            if (s[j] == ' ') {
                s[i--] = '0';
                s[i--] = '2';
                s[i--] = '%';
            } else {
                s[i--] = s[j];
            }
        }

        return s;
    }
};
```
# 剑指 Offer 58 - II. 左旋转字符串
### 方法1：库函数做法
```
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        while (n--) {
            auto x = s[0];
            s.erase(s.begin(), s.begin() + 1);
            s += x;
        }

        return s;
    }
};
```
### 方法2 分段处理
```
class Solution {
public:
    string reverseLeftWords(string s, int k) {
        int n = s.size();
        k %= n;
        string s1 = s.substr(0, k);
        string s2 = s.substr(k);
        string res = s2 + s1;

        return res;
    }
};
```
### 方法3 deque
```
class Solution {
public:
    string reverseLeftWords(string str, int n) {
        deque<char> dq;
        for (auto x : str)
            dq.push_back(x);
        for (int i = 0; i < n; ++i) {
            auto t = dq.front();
            dq.pop_front();
            dq.push_back(t);
        }
        string s;
        for (int i = 0; i < dq.size(); ++i) {
            s += dq[i];
        }
        return s;
    }
};
```
### 方法4 原地反转
```
class Solution {
public:
    string reverseLeftWords(string s, int n) {
        reverse(s.begin(), s.begin() + n);
        reverse(s.begin() + n, s.end());
        reverse(s.begin(), s.end());

        return s;
    }
};
```
# 28. 找出字符串中第一个匹配项的下标
### 双指针做法 KMP暂未实现
```
class Solution {
public:
    int strStr(string s, string t) {
        int ls = s.size(), lt = t.size();
        int i = 0;
        while (i < ls) {
            if (s[i] == t[0]) {
                int j = 0, k = i;
                while (k < ls && s[k] == t[j]) {
                    k++;
                    j++;
                }

                if (j == lt) return i;
            }
            i++;
        }

        return -1;
    }
};
```


# 150. 逆波兰表达式求值
```
class Solution {
public:
    stack<int> num;
    stack<char> op;
    vector<string> v{"+", "-", "*", "/"};

    int evalRPN(vector<string>& tokens) {
        for (auto x : tokens) {
            if (find(v.begin(), v.end(), x) != v.end()) {
                int a = num.top(); num.pop();
                int b = num.top(); num.pop();
                if (x == "+") {
                    num.push(a + b);
                } else if (x == "-") {
                    num.push(b - a);
                } else if (x == "*") {
                    num.push(a * b);
                } else if (x == "/") {
                    num.push(b / a);
                }
            } else {
                num.push(stoi(x));
            }
        }

        return num.top();
    }
};
```

# 102. 二叉树的层序遍历
```
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if (!root) return res;
        queue<TreeNode*> q;
        q.push(root);
        
        while (q.size()) {
            int len = q.size();
            vector<int> temp;
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                temp.push_back(node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            res.push_back(temp);
        }

        return res;
    }
};
```
# 107. 二叉树的层次遍历 II
```
class Solution {
public:
    vector<vector<int>> levelOrderBottom(TreeNode* root) {
        vector<vector<int>> ans, res;
        if (!root) return ans;
        queue<TreeNode*> q;
        q.push(root);

        while (q.size()) {
            int len = q.size();
            vector<int> t;
            for (int i = 0; i < len; i++) {
                auto h = q.front();
                q.pop();
                t.push_back(h->val);
                if (h->left) q.push(h->left);
                if (h->right) q.push(h->right);
            }
            ans.push_back(t);
        }

        for (int i = ans.size() - 1; i >= 0; i--)
            res.push_back(ans[i]);

        return res;
    }
};
```

# 429. N叉树的层序遍历
### BFS
```
class Solution {
public:
    vector<vector<int>> levelOrder(Node* root) {
        vector<vector<int>> ans;
        if (!root) return ans;
        queue<Node*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            vector<int> temp;
            for (int i = 0; i < len; i++) {
                auto nd = q.front();
                q.pop();
                temp.push_back(nd->val);
                for (auto x : nd->children) {
                    if (x) q.push(x);
                }
            }
            ans.push_back(temp);
        }

        return ans;
    }
};
```
# 515. 在每个树行中找最大值
### BFS
```
class Solution {
public:
    vector<int> largestValues(TreeNode* root) {
        vector<int> ans;
        if (!root) return ans;
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            int t =INT_MIN;
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                t = max(t, node->val);
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            ans.push_back(t);
        }

        return ans;
    }
};
```
# 637. 二叉树的层平均值
### BFS
```
class Solution {
public:
    vector<double> averageOfLevels(TreeNode* root) {
        vector<double> ans;
        if (!root) return ans;
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            double s = 0;
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                s += node->val;
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
            }
            ans.push_back(s / len);
        }

        return ans;
    }
};
```
# 116. 填充每个节点的下一个右侧节点指针
### 方法1 BFS
```
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        root->next = NULL;
        queue<Node*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            vector<Node*> v;
            for (int i = 0; i < len; i++) {
                auto t = q.front();
                q.pop();
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
                v.push_back(t);
            }

            for (int i = 0; i < len; i++) {
                if (i + 1 < len) {
                    v[i]->next = v[i + 1];
                } else {
                    v[i]->next = NULL;
                }
            }
        }

        return root;
    }
};
```

### 方法2 BFS
```
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        queue<Node*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto t = q.front();
                q.pop();
                if (i != len - 1) {
                    auto tt = q.front();
                    t->next = tt;
                } else {
                    t->next = NULL;
                }
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
        }

        return root;
    }
};
```

### 方法3 BFS O(1) 只适用于完全二叉树
```
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return NULL;
        auto ans = root;
        while (root->left) {
            auto p = root;
            for (; p; p = p->next) {
                p->left->next = p->right;
                if (p->next) {
                    p->right->next = p->next->left;
                }
            }

            root = root->left;
        }

        return ans;
    }
};
```
### 方法4 BFS O(1) 适用于任意二叉树
```
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        auto cur = root;

        while (cur) {
            auto h = new Node();
            auto t = h;
            for (auto p = cur; p; p = p->next) {
                if (p->left) {
                    t->next = p->left;
                    t = t->next;
                }

                if (p->right) {
                    t->next = p->right;
                    t = t->next;
                }
            }
            cur = h->next;
        }

        return root;
    }
};
```
# 117. 填充每个节点的下一个右侧节点指针 II
### 方法1：BFS
```
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        queue<Node*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto t = q.front();
                q.pop();
                if (i != len - 1) {
                    auto tt = q.front();
                    t->next = tt;
                } else {
                    t->next = NULL;
                }
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
        }

        return root;
    }
};
```

### 方法2：BFS O(1)空间
```
class Solution {
public:
    Node* connect(Node* root) {
        if (!root) return root;
        auto cur = root;

        while (cur) {
            auto h = new Node();
            auto t = h;
            for (auto p = cur; p; p = p->next) {
                if (p->left) {
                    t->next = p->left;
                    t = t->next;
                }

                if (p->right) {
                    t->next = p->right;
                    t = t->next;
                }
            }
            cur = h->next;
        }

        return root;
    }
};
```

# 100. 相同的树
### 方法1 DFS
```
class Solution {
public:
    bool isSameTree(TreeNode* p, TreeNode* q) {
        if (!p && !q) return true;
        if (!p && q) return false;
        if (p && !q) return false;
        if (p->val != q->val) return false;
        return isSameTree(p->left, q->left) && isSameTree(p->right, q->right);
    }
};
```
### 方法2：BFS变形
```
class Solution {
public:
    bool isSameTree(TreeNode* root0, TreeNode* root1) {
        queue<TreeNode*> q;
        q.push(root0);
        q.push(root1);
        while (q.size()) {
            auto n0 = q.front(); q.pop();
            auto n1 = q.front(); q.pop();

            if (!n0 && !n1) continue;
            if (!(n0 && n1)) return false;
            if (n0->val != n1->val) return false;
            q.push(n0->left);
            q.push(n1->left);
            q.push(n0->right);
            q.push(n1->right);
        } 

        return true;
    }
};
```
# 559. N叉树的最大深度
### 方法1：bfs
```
class Solution {
public:
    int maxDepth(Node* root) {
        queue<Node*> q;
        if (!root) return 0;
        int d = 0;
        q.push(root);
        while (q.size()) {
            d++;
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto nd = q.front();
                q.pop();
                for (auto x : nd->children) {
                    if (x) q.push(x);
                }
            }
        }
        return d;
    }
};
```
### 方法2：DFS
```

class Solution {
public:
    int maxDepth(Node* root) {
        if (!root) return 0;
        int r = 0;
        for (auto x : root->children) {
            r = max(r, maxDepth(x));
        }

        return r + 1;
    }
};
```

# 222. 完全二叉树的节点个数
### 方法1：BFS
```
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (!root) return 0;
        queue<TreeNode*> q;
        q.push(root);

        int cnt = 0;
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto t = q.front(); q.pop();
                cnt++;
                if (t->left) q.push(t->left);
                if (t->right) q.push(t->right);
            }
        }

        return cnt;

    }
};
```
# 方法2：DFS 普通二叉树，未利用完全二叉树的性质
```
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (!root) return 0;
        int l = countNodes(root->left);
        int r = countNodes(root->right);

        return l + r + 1;
    }
};
```

### 方法3：DFS 利用完全二叉树的性质
```
class Solution {
public:
    int countNodes(TreeNode* root) {
        if (!root) return 0;
        int dl = 1, dr = 1;
        auto p = root->left;
        while (p) {
            dl++;
            p = p->left;
        }

        p = root->right;
        while (p) {
            dr++;
            p = p->right;
        }
        if (dr == dl) {
            return (1 << dr) - 1;
        }

        return countNodes(root->left) + countNodes(root->right) + 1;
    }
};
```
# 257. 二叉树的所有路径
### 方法1：DFS
```
class Solution {
public:
    vector<string> ans;

    void dfs(TreeNode* root, string s) {
        if (!root->left && !root->right) {
            s.pop_back();
            s.pop_back();
            ans.push_back(s);
            return;
        }
        if (root->left)
            dfs(root->left, s + to_string(root->left->val) + "->");
        if (root->right)
            dfs(root->right, s + to_string(root->right->val) + "->");
    }
    vector<string> binaryTreePaths(TreeNode* root) {
        if (!root) return ans;
        dfs(root, to_string(root->val) + "->");
        return ans;
    }
};
```
# 404. 左叶子之和
### 方法1:BFS
```
class Solution {
public:
    int sumOfLeftLeaves(TreeNode* root) {
        int ans = 0;
        if (!root) return 0;
        // if (root->left == nullptr && root->right == nullptr) return 0;
        queue<TreeNode*> q;
        q.push(root);
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);

                auto p = node;
                if (p && p->left) {
                    p = p->left;
                    if (p->left == nullptr && p->right == nullptr) {
                        ans += p->val;
                    }
                }
            }
        }

        return ans;
    }
};
```
### 方法2:DFS
```
class Solution {
public:
    int ans;
    int sumOfLeftLeaves(TreeNode* root) {
        ans = 0;
        dfs(root);
    
        return ans;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        if (root && root->left) {
            auto p = root->left;
            if (!p->left && !p->right) ans += p->val;
        }
        dfs(root->right);
    }
};
```
### 方法3:DFS
```
class Solution {
public:
    int ans = 0;
    int sumOfLeftLeaves(TreeNode* root) {
        dfs(root, false);

        return ans;
    }

    void dfs(TreeNode* root, bool isLeft) {
        if (!root) return;
        if (isLeft && root->left == nullptr && root->right == nullptr) ans += root->val;
        if (root->left) dfs(root->left, true);
        if (root->right) dfs(root->right, false);
    }
};
```
# 513. 找树左下角的值
### 方法1：BFS
```
class Solution {
public:
    int findBottomLeftValue(TreeNode* root) {
        queue<TreeNode*> q;
        q.push(root);
        int ans;
        while (q.size()) {
            int len = q.size();
            for (int i = 0; i < len; i++) {
                auto node = q.front();
                q.pop();
                if (node->left) q.push(node->left);
                if (node->right) q.push(node->right);
                if (i == 0) ans = node->val;
            }
        }

        return ans;
    }
};
```
### 方法2：DFS
```
class Solution {
public:
    int ans;
    vector<bool> st;
    int findBottomLeftValue(TreeNode* root) {
        st = vector<bool>(1100);
        dfs(root, 0);

        return ans;
    }

    void dfs(TreeNode* root, int d) {
        if (!root) return;
        if (!st[d]) {
            ans = root->val;
            st[d] = true;
        }
        dfs(root->left, d + 1);
        dfs(root->right, d + 1);
    }
};
```
### 方法3：DFS
```
class Solution {
public:
    int ans, maxd = 0;
    int findBottomLeftValue(TreeNode* root) {
        dfs(root, 1);

        return ans;
    }

    void dfs(TreeNode* root, int d) {
        if (!root) return;

        if (d > maxd) {
            maxd = d;
            ans = root->val;
        }

        dfs(root->left, d + 1);
        dfs(root->right, d + 1);
    }
};
```


# 654. 最大二叉树
### 方法1：DFS
```
class Solution {
public:
    TreeNode* constructMaximumBinaryTree(vector<int>& nums) {
        return buildTree(nums, 0, nums.size() - 1);
    }

    TreeNode* buildTree(vector<int>& nums, int l, int r) {
        if (l > r) return nullptr;
        int v = getMaxIndex(nums, l, r);
        TreeNode* root = new TreeNode(nums[v]);
        root->left = buildTree(nums, l, v - 1);
        root->right = buildTree(nums, v + 1, r);

        return root;
    }

    int getMaxIndex(vector<int>& nums, int l, int r) {
        int ans = l;
        for (int i = l; i <= r; i++) {
            if (nums[i] > nums[ans]) {
                ans = i;
            }
        }

        return ans;
    }
};
```

# 617. 合并二叉树
### 方法1：DFS 自己的写法
```
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        TreeNode* r = nullptr;
        if (!root1 && !root2) return nullptr;
        if (!root1 && root2) {
            r = new TreeNode(root2->val);
            r->left = mergeTrees(nullptr, root2->left);
            r->right = mergeTrees(nullptr, root2->right);
        }
        if (root1 && !root2) {
            r = new TreeNode(root1->val);
            r->left = mergeTrees(root1->left, nullptr);
            r->right = mergeTrees(root1->right, nullptr);
        }
        if (root1 && root2) {
            r = new TreeNode(root1->val + root2->val);
            r->left = mergeTrees(root1->left, root2->left);
            r->right = mergeTrees(root1->right, root2->right);
        }

        return r;
    }
};
```
### 方法2：DFS 简洁的写法
```
class Solution {
public:
    TreeNode* mergeTrees(TreeNode* root1, TreeNode* root2) {
        auto root = new TreeNode();
        if (!root1) return root2;
        if (!root2) return root1;

        root->val = root1->val + root2->val;
        root->left = mergeTrees(root1->left, root2->left);
        root->right = mergeTrees(root1->right, root2->right);

        return root;
    }
};
```

# 700. 二叉搜索树中的搜索
### 方法1：BST 迭代
```
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        while (root) {
            if (root->val == val) return root;
            else if (root->val > val) {
                root = root->left;
            } else {
                root = root->right;
            }
        }

        return root;
    }
};
```
### 方法2：BST递归
```
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if (!root || root->val == val) return root;

        if (root->val < val) return searchBST(root->right, val);
        if (root->val > val) return searchBST(root->left, val);

        return root;
    }
};
```
### 方法3：常规DFS
```
class Solution {
public:
    TreeNode* searchBST(TreeNode* root, int val) {
        if (!root) return root;
        if (root->val == val) return root;
        auto l = searchBST(root->left, val);
        auto r = searchBST(root->right, val);

        if (!l) return r;
        return l;
    }
};
```
# 530. 二叉搜索树的最小绝对差
### 方法1：中序遍历先存储结果再判断
```
class Solution {
public:
    vector<int> vec;
    int getMinimumDifference(TreeNode* root) {
        dfs(root);

        int r = INT_MAX;
        for (int i = 1; i < vec.size(); i++) {
            r = min(r, abs(vec[i] - vec[i - 1]));
        }

        return r;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        vec.push_back(root->val);
        dfs(root->right);

    }
};
```
### 方法2：中序遍历DFS记录前一个节点然后判断
```
class Solution {
public:
    TreeNode* pre;
    int ans;
    int getMinimumDifference(TreeNode* root) {
        pre = nullptr;
        ans = INT_MAX;
        dfs(root);

        return ans;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        if (pre) {
            ans = min(ans, root->val - pre->val);
        }

        pre = root;
        dfs(root->right);
    }
};
```
### 方法3：中序遍历迭代记录前一个节点然后判断
```
class Solution {
public:
    int getMinimumDifference(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* pre = nullptr;
        int ans = INT_MAX;
        while (st.size() || root) {
            if (root) {
                st.push(root);
                root = root->left;
            } else {
                auto node = st.top();
                st.pop();
                if (pre) {
                    ans = min(ans, node->val - pre->val);
                }
                pre = node;
                root = node->right;
            }
        }
        return ans;
    }
};
```

# 783. 二叉搜索树节点最小距离
### 方法1：中序遍历先存储结果再判断
```
class Solution {
public:
    vector<int> vec;
    int minDiffInBST(TreeNode* root) {
        dfs(root);

        int r = INT_MAX;
        for (int i = 1; i < vec.size(); i++) {
            r = min(r, abs(vec[i] - vec[i - 1]));
        }

        return r;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        vec.push_back(root->val);
        dfs(root->right);

    }
};
```
### 方法2：中序遍历DFS记录前一个节点然后判断
```
class Solution {
public:
    TreeNode* pre;
    int ans;
    int minDiffInBST(TreeNode* root) {
        pre = nullptr;
        ans = INT_MAX;
        dfs(root);

        return ans;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        if (pre) {
            ans = min(ans, root->val - pre->val);
        }

        pre = root;
        dfs(root->right);
    }
};
```
### 方法3：中序遍历迭代记录前一个节点然后判断
```
class Solution {
public:
    int minDiffInBST(TreeNode* root) {
        stack<TreeNode*> st;
        TreeNode* pre = nullptr;
        int ans = INT_MAX;
        while (st.size() || root) {
            if (root) {
                st.push(root);
                root = root->left;
            } else {
                auto node = st.top();
                st.pop();
                if (pre) {
                    ans = min(ans, node->val - pre->val);
                }
                pre = node;
                root = node->right;
            }
        }
        return ans;
    }
};
```

# 501. 二叉搜索树中的众数
### 方法1：两次DFS
```
class Solution {
public:
    vector<int> ans;
    int cnt, maxCnt;
    TreeNode* pre;
    vector<int> findMode(TreeNode* root) {
        cnt = 1, maxCnt = 1;
        pre = nullptr;
        dfs0(root);
        pre = nullptr;
        dfs(root);
        return ans;
    }
    
    void dfs0(TreeNode* root) {
        if (!root) return;
        dfs0(root->left);
        if (pre) {
            if (pre->val == root->val) {
                cnt++;
            } else {
                cnt = 1;     
            }
        } else {
            cnt = 1;
        }

        if (cnt > maxCnt) {
            maxCnt = cnt;
        }
        pre = root;
        dfs0(root->right);
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        if (pre) {
            if (pre->val == root->val) {
                cnt++;
            } else {
                cnt = 1;     
            }
        } else {
            cnt = 1;
        }

        if (cnt == maxCnt) {
            ans.push_back(root->val);
        }
        pre = root;
        dfs(root->right);
    }
};
```
### 方法2：一次DFS
```
class Solution {
public:
    vector<int> ans;
    int cnt, maxCnt;
    TreeNode* pre;
    vector<int> findMode(TreeNode* root) {
        cnt = 1, maxCnt = 1;
        pre = nullptr;
        dfs(root);
        return ans;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->left);
        if (pre) {
            if (pre->val == root->val) {
                cnt++;
            } else {
                cnt = 1;     
            }
        } else {
            cnt = 1;
        }

        if (cnt == maxCnt) {
            ans.push_back(root->val);
        }
        if (cnt > maxCnt) {
            maxCnt = cnt;
            ans.clear();
            ans.push_back(root->val);
        }
        pre = root;
        dfs(root->right);
    }
};
```
### 方法3：迭代的中序遍历实现
```
class Solution {
public:
    vector<int> findMode(TreeNode* root) {
        vector<int> ans;
        int cnt = 1, maxCnt = 1;
        TreeNode* pre = nullptr;
        stack<TreeNode*> st;
        while (root || st.size()) {
            if (root) {
                st.push(root);
                root = root->left;
            } else {
                auto node = st.top();
                st.pop();

                if (pre) {
                    if (pre->val == node->val) {
                        cnt++;
                    } else {
                        cnt = 1;
                    }
                } else {
                    cnt = 1;
                }

                if (cnt == maxCnt) {
                    ans.push_back(node->val);
                }

                if (cnt > maxCnt) {
                    // cout << "cnt = " << cnt << ", maxCnt = " << maxCnt << endl;
                    maxCnt = cnt;
                    ans.clear();
                    ans.push_back(node->val);
                }

                pre = node;
                root = node->right;
            }
        }

        return ans;
    }
};
```
# 701. 二叉搜索树中的插入操作
### 方法1：dfs
```
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (!root) {
            TreeNode* node = new TreeNode(val);

            return node;
        }

        if (root->val > val) root->left = insertIntoBST(root->left, val);
        if (root->val < val) root->right = insertIntoBST(root->right, val);

        return root;
    }
};
```
### 方法2：迭代
```
class Solution {
public:
    TreeNode* insertIntoBST(TreeNode* root, int val) {
        if (!root) {
            TreeNode* node = new TreeNode(val);
            return node;
        }
        TreeNode* ans = root;
        TreeNode* pre = root;

        while (root) {
            if (root->val > val) {
                pre = root;
                root = root->left;
                if (!root) {
                    TreeNode* node = new TreeNode(val);
                    pre->left = node;
                    break;
                }
            } else if (root->val < val) {
                pre = root;
                root = root->right;
                if (!root) {
                    TreeNode* node = new TreeNode(val);
                    pre->right = node;
                    break;
                }
            }
        }
        
        return ans;
    }
};
```
# 669. 修剪二叉搜索树
### 方法1：dfs
```
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if (!root) return nullptr;
        if (root->val < low) {
            return trimBST(root->right, low, high);
        } else if (root->val > high) {
            return trimBST(root->left, low, high);
        } 

        root->left = trimBST(root->left, low, high);
        root->right = trimBST(root->right, low, high);

        return root;
    }
};
```

### 方法2：迭代写法
```
class Solution {
public:
    TreeNode* trimBST(TreeNode* root, int low, int high) {
        if (!root) return root;

        while (root && (root->val < low || root->val > high)) {
            if (root->val < low) root = root->right;
            else root = root->left;
        }

        auto p = root;
        while (p) {
            while (p->left && p->left->val < low) {
                p->left = p->left->right;
            }
            p = p->left;
        }

        p = root;

        while (p) {
            while (p->right && p->right->val > high) {
                p->right = p->right->left;
            }
            p = p->right;
        }
        return root;
    }
};
```
# 108. 将有序数组转换为二叉搜索树
### 方法1：二叉搜索树DFS，一定要想到二分
```
class Solution {
public:
    TreeNode* sortedArrayToBST(vector<int>& nums) {
        return dfs(nums, 0, nums.size() - 1);
    }

    TreeNode* dfs(vector<int>& nums, int l, int r) {
        if (l > r) return nullptr;
        int mid = l + r >> 1;
        TreeNode* root = new TreeNode(nums[mid]);
        root->left = dfs(nums, l, mid - 1);
        root->right = dfs(nums, mid + 1, r);

        return root;
    }
};
```
# 538. 把二叉搜索树转换为累加树
### 方法1：dfs
```
class Solution {
public:
    TreeNode* pre;
    TreeNode* convertBST(TreeNode* root) {
        pre = nullptr;
    
        dfs(root);
        return root;
    }

    void dfs(TreeNode* root) {
        if (!root) return;
        dfs(root->right);
        if (pre) {
            root->val += pre->val;
        }
        pre = root;
        dfs(root->left);
    }
};
```
### 方法2：迭代中序遍历
```
class Solution {
public:
    TreeNode* convertBST(TreeNode* root) {
        TreeNode* pre = nullptr;
        stack<TreeNode*> st;
        auto ans = root;
        while (root || st.size()) {
            if (root) {
                st.push(root);
                root = root->right;
            } else {
                auto node = st.top();
                st.pop();
                if (pre) {
                    node->val += pre->val;
                }

                pre = node;
                root = node->left;
            }
        }

        return ans;
    }
};
```
# 1038. 从二叉搜索树到更大和树
### 方法1：DFS
```
class Solution {
public:
    TreeNode* pre = nullptr;
    TreeNode* bstToGst(TreeNode* root) {
        if (!root) return root;
        bstToGst(root->right);
        if (pre) {
            root->val += pre->val;
        }
        pre = root;
        bstToGst(root->left);

        return root;
    }
};
```

### 方法2：迭代
```
class Solution {
public:
    TreeNode* bstToGst(TreeNode* root) {
        TreeNode* pre = nullptr;
        stack<TreeNode*> st;
        auto ans = root;
        while (root || st.size()) {
            if (root) {
                st.push(root);
                root = root->right;
            } else {
                auto node = st.top();
                st.pop();
                if (pre) {
                    node->val += pre->val;
                }

                pre = node;
                root = node->left;
            }
        }

        return ans;
    }
};
```
# 77. 组合
### 方法1：dfs
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> combine(int n, int k) {
        dfs(n, 1, k);

        return ans;
    }

    void dfs(int n, int start, int k) {
        if (path.size() == k) {
            ans.push_back(path);
            return;
        }

        for (int i = start; i <= n; i++) {
            path.push_back(i);
            dfs(n, i + 1, k);
            path.pop_back();
        }
    }
};
```
### 方法2：dfs剪枝
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> combine(int n, int k) {
        dfs(n, 1, k);

        return ans;
    }

    void dfs(int n, int start, int k) {
        if (path.size() == k) {
            ans.push_back(path);
            return;
        }

        for (int i = start; k - path.size() <= n - i + 1; i++) {
            path.push_back(i);
            dfs(n, i + 1, k);
            path.pop_back();
        }
    }
};
```
# 216. 组合总和 III
### 方法1：DFS
```
class Solution {
public:
    vector<int> path;
    vector<vector<int>> ans;
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(k, n, 1, 0);

        return ans;
    }

    void dfs(int k, int n, int u, int s) {
        if (path.size() > k) return;
        if (s > n) return;
        if (s == n) {
            if (path.size() < k) return;
            if (path.size() == k) {
                ans.push_back(path);
                return;
            }
        }

        for (int i = u; i <= 9; i++) {
            path.push_back(i);
            dfs(k, n, i + 1, s + i);
            path.pop_back();
        }
    }
};
```
### 方法2：DFS剪枝
```
class Solution {
public:
    vector<int> path;
    vector<vector<int>> ans;
    vector<vector<int>> combinationSum3(int k, int n) {
        dfs(k, n, 1, 0);

        return ans;
    }

    void dfs(int k, int n, int u, int s) {
        if (path.size() > k) return;
        if (s > n) return;
        if (s == n) {
            if (path.size() < k) return;
            if (path.size() == k) {
                ans.push_back(path);
                return;
            }
        }

        for (int i = u; i <= 9 && s + i <= n; i++) {
            path.push_back(i);
            dfs(k, n, i + 1, s + i);
            path.pop_back();
        }
    }
};
```

# 17. 电话号码的字母组合
```
class Solution {
public:
    string gMap[10] = {"", "", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
    vector<string> ans;
    vector<string> letterCombinations(string digits) {
        if (digits.size() == 0) return ans;
        dfs(digits, 0, "");

        return ans;
    }

    void dfs(string& digits, int u, string s) {
        if (u == digits.size()) {
            ans.push_back(s);
            return;
        }

        int id = digits[u] - '0';
        string t = gMap[id];
        for (int i = 0; i < t.size(); i++) {
            dfs(digits, u + 1, s + t[i]);
        }
    }
};
```

# 131. 分割回文串
### 方法1：DFS
```
class Solution {
public:
    vector<vector<string>> ans;
    vector<string> path;
    vector<vector<string>> partition(string s) {
        dfs(s, 0);

        return ans;
    }

    void dfs(string& s, int start) {
        if (start == s.size()) {
            ans.push_back(path);
            return;
        }
        for (int i = start; i < s.size(); i++) {
            string temp = s.substr(start, i - start + 1);
            if (isValid(temp)) {
                path.push_back(temp);
                dfs(s, i + 1);
                path.pop_back();
            }
        }
    }

    bool isValid(string &s) {
        int l = 0, r = s.size() - 1;
        while (l < r) {
            if (s[l] == s[r]) {
                l++;
                r--;
            } else {
                return false;
            }
        }

        return true;
    }
};
```

### 方法2：DFS + DP优化
```
class Solution {
public:
    vector<string> path;
    vector<vector<string>> ans;
    vector<vector<bool>> f;
    vector<vector<string>> partition(string s) {
        int n = s.size();
        f = vector<vector<bool>>(n, vector<bool>(n));
        for (int j = 0; j < n; j++) {
            for (int i = 0; i <= j; i++) {
                if (i == j) f[i][j] = true;
                else if (s[i] == s[j]) {
                    if (j - i == 1) f[i][j] = true;
                    else if (f[i + 1][j - 1]) f[i][j] = true;
                }
            }
        }

        dfs(s, 0);

        return ans;
    }

    void dfs(string& s, int u) {
        if (u == s.size()) {
            ans.push_back(path);
        } else {
            for (int i = u; i < s.size(); i++) {
                if (f[u][i]) {
                    path.push_back(s.substr(u, i - u + 1));
                    dfs(s, i + 1);
                    path.pop_back();
                }
            }
        }
    }
};
```
# 90. 子集 II
### 方法1：DFS，去重
```
class Solution {
public:
    vector<int> path;
    vector<vector<int>> ans;
    vector<bool> st;

    void dfs(vector<int>& nums, int u) {
        ans.push_back(path);

        for (int i = u; i < nums.size(); i++) {
            if (i && nums[i - 1] == nums[i] && !st[i - 1]) {
                continue;
            }

            path.push_back(nums[i]);
            st[i] = true;
            dfs(nums, i + 1);
            st[i] = false;
            path.pop_back();
        }
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        st = vector<bool>(nums.size());
        dfs(nums, 0);

        return ans;
    }
};
```

### 方法2：DFS，另一种去重方法
```
class Solution {
public:
    vector<int> path;
    vector<vector<int>> ans;
    vector<bool> st;

    void dfs(vector<int>& nums, int u) {
        ans.push_back(path);

        for (int i = u; i < nums.size(); i++) {
            if (i > u && nums[i - 1] == nums[i]) {
                continue;
            }

            path.push_back(nums[i]);
            dfs(nums, i + 1);
            path.pop_back();
        }
    }

    vector<vector<int>> subsetsWithDup(vector<int>& nums) {
        sort(nums.begin(), nums.end());
        dfs(nums, 0);

        return ans;
    }
};
```
# 491. 递增子序列
### 方法1：DFS + 库函数去重
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> findSubsequences(vector<int>& nums) {
        dfs(nums, 0);
        sort(ans.begin(), ans.end());
        ans.erase(unique(ans.begin(), ans.end()), ans.end());
        return ans;
    }

    void dfs(vector<int>& nums, int u) {
        if (path.size() >= 2) {
            ans.push_back(path);
            // return;
        }

        if (u == nums.size()) return;

        for (int i = u; i < nums.size(); i++) {
            if (path.empty() || nums[i] >= path.back()) {
                path.push_back(nums[i]);
                dfs(nums, i + 1);
                path.pop_back();
            }
        }
    }
};
```

### 方法2：DFS + set去重
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> findSubsequences(vector<int>& nums) {
        dfs(nums, 0);
        // sort(ans.begin(), ans.end());
        // ans.erase(unique(ans.begin(), ans.end()), ans.end());
        return ans;
    }

    void dfs(vector<int>& nums, int u) {
        if (path.size() >= 2) {
            ans.push_back(path);
        }

        if (u == nums.size()) return;
        unordered_set<int> uset;
        for (int i = u; i < nums.size(); i++) {
            if (uset.find(nums[i]) != uset.end()) continue;
            if (path.empty() || nums[i] >= path.back()) {
                path.push_back(nums[i]);
                uset.insert(nums[i]);
                dfs(nums, i + 1);
                path.pop_back();
            }
        }
    }
};
```


### 方法3：DFS + 数组去重
```
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> findSubsequences(vector<int>& nums) {
        dfs(nums, 0);
        // sort(ans.begin(), ans.end());
        // ans.erase(unique(ans.begin(), ans.end()), ans.end());
        return ans;
    }

    void dfs(vector<int>& nums, int u) {
        if (path.size() >= 2) {
            ans.push_back(path);
        }

        if (u == nums.size()) return;

        int h[300] = {0};
        for (int i = u; i < nums.size(); i++) {
            if (h[nums[i] + 100] == 1) continue;
            if (path.empty() || nums[i] >= path.back()) {
                path.push_back(nums[i]);
                h[nums[i] + 100] = 1;
                dfs(nums, i + 1);
                path.pop_back();
            }
        }
    }
};
```

# 332. 重新安排行程
### 方法1：欧拉回路
```
class Solution {
public:
    unordered_map<string, multiset<string>> g;
    vector<string> ans;
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        for (auto e : tickets) {
            g[e[0]].insert(e[1]);
        }

        dfs("JFK");
        reverse(ans.begin(), ans.end());

        return ans;
    }

    
    void dfs(string s) {
        while (g[s].size()) {
            auto x = *g[s].begin();
            g[s].erase(g[s].begin());
            dfs(x);
        }

        ans.push_back(s);
    }
};
```
### 方法2：回溯，更好理解
```
class Solution {
public:
    vector<string> result;
    unordered_map<string, map<string, int>> g;
    int n;
    vector<string> findItinerary(vector<vector<string>>& tickets) {
        n = tickets.size();
        for (auto x : tickets) {
            g[x[0]][x[1]]++;
        }

        result.push_back("JFK");
        dfs();
        return result;
    }

    bool dfs() {
        if (result.size() == n + 1) {
            return true;
        }
        auto x = result.back();
        for (auto &t : g[x]) {
            if (t.second > 0) {
                result.push_back(t.first);
                t.second--;
                if (dfs()) return true;
                result.pop_back();
                t.second++;
            }
        }

        return false;
    }
};
```

# 51. N 皇后
### 方法1：DFS 按行枚举
```
class Solution {
public:
    vector<vector<string>> ans;
    vector<bool> row, col, up, down;
    vector<string> g;
    int n;
    vector<vector<string>> solveNQueens(int _n) {
        n = _n;
        string str(n, '.');
        // cout << "str = " << str << endl;
        g = vector<string>(n, str);
        row = vector<bool>(n);
        col = vector<bool>(n);
        up = vector<bool>(2 * n);
        down = vector<bool>(2 * n);
        dfs(0);

        return ans;
    }

    // u是行数
    void dfs(int u) {
        if (u == n) {
            ans.push_back(g);
            return;
        }
        // i是列数
        for (int i = 0; i < n; i++) {
            if (!row[u] && !col[i] && !up[u + i] && !down[u - i + n]) {
                row[u] = col[i] = up[u + i]= down[u - i + n] = true;
                g[u][i] = 'Q';
                dfs(u + 1);
                g[u][i] = '.';
                row[u] = col[i] = up[u + i]= down[u - i + n] = false;
            }
        }
    }
};
```
# 455. 分发饼干
### 方法1：贪心
```
class Solution {
public:
    int findContentChildren(vector<int>& g, vector<int>& s) {
        int ans = 0;
        int l = 0, r = 0;
        sort(g.begin(), g.end());
        sort(s.begin(), s.end());
        while (l < g.size() && r < s.size()) {
            if (g[l] <= s[r]) {
                l++;
                r++;
                ans++;
            } else {
                r++;
            }
        }

        return ans;
    }
};
```
# 376. 摆动序列
### 方法1：DP
```
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int n = nums.size();
        vector<vector<int>> dp(n + 1, vector<int>(2));
        dp[0][0] = dp[0][1] = 1;
        int res = 1;

        for (int i = 1; i < n; i++) {
            if (nums[i] - nums[i - 1] > 0) {
                dp[i][1] = dp[i - 1][0] + 1;
                dp[i][0] = dp[i - 1][0];
            } else if (nums[i] - nums[i - 1] < 0) {
                dp[i][0] = dp[i - 1][1] + 1;
                dp[i][1] = dp[i - 1][1];
            } else {
                dp[i][1] = dp[i - 1][1];
                dp[i][0] = dp[i - 1][0];
            }
            res = max(res, dp[i][0]);
            res = max(res, dp[i][1]);
        }

        return res; 
    }
};
```

### 方法2：贪心
```
class Solution {
public:
    int wiggleMaxLength(vector<int>& nums) {
        int res = 1;
        int pre = 0, cur = 0;
        for (int i = 1; i < nums.size(); i++) {
            cur = nums[i] - nums[i - 1];
            if ((cur > 0 && pre <= 0) || (cur < 0 && pre >= 0)) {
                res++;
                pre = cur;
            }
            // pre = cur;
        } 
        return res;
    }
};
```

# 1005. K 次取反后最大化的数组和
### 方法1：贪心自己的实现
```
class Solution {
public:
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end());
        while (k--) {
            if (nums[0] < 0) {
                nums[0] = -nums[0];
                sort(nums.begin(), nums.end());
            } else if (nums[0] == 0) {
                break;
            } else if (nums[0] > 0) {
                nums[0] = -nums[0];
                sort(nums.begin(), nums.end());
            }
        }

        // for (auto x : nums) {
        //     cout << x << " ";
        // }
        // cout << endl;

        int ans = accumulate(nums.begin(), nums.end(), 0);

        return ans;
    }
};
```

### 方法2：贪心
```
class Solution {
public:
    int largestSumAfterKNegations(vector<int>& nums, int k) {
        sort(nums.begin(), nums.end(), [](int x, int y) {
            return abs(x) > abs(y);
        });

        for (int i = 0; i < nums.size(); i++) {
            if (nums[i] < 0 && k > 0) {
                nums[i] = -nums[i];
                k--;
            }
        }
        if (k % 2) nums[nums.size() - 1] = -nums[nums.size() - 1];
        int ans = accumulate(nums.begin(), nums.end(), 0);
        return ans;
    }
};
```
# 860. 柠檬水找零
### 方法1：贪心
```
class Solution {
public:
    bool lemonadeChange(vector<int>& bills) {
        int five = 0, ten = 0, twenty = 0;
        for (auto x : bills) {
            if (x == 5) {
                five++;
            } else if (x == 10) {
                ten++;
                if (five > 0) five--;
                else return false;
            } else if (x == 20) {
                if (ten > 0 && five > 0) {
                    ten--;
                    five--;
                } else if (five >= 3) {
                    five -= 3;
                } else {
                    return false;
                }
            }
        }

        return true;
    }
};
```
# 406. 根据身高重建队列
### 方法1：贪心 + 排序
```
class Solution {
public:
    static bool comp(vector<int> a, vector<int> b) {
        if (a[0] == b[0]) return a[1] < b[1];
        return a[0] > b[0];
    }
    vector<vector<int>> reconstructQueue(vector<vector<int>>& people) {
        sort(people.begin(), people.end(), comp);

        vector<vector<int>> ans;
        for (auto x : people) {
            ans.insert(ans.begin() + x[1], x);
        }

        return ans;
    }
};
```
# 452. 用最少数量的箭引爆气球
### 方法1：贪心，按右端点排序
```
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        sort(points.begin(), points.end(),
        [](vector<int>& a, vector<int>& b) {
            return a[1] < b[1];
        });

        int cnt = 1;
        int py = points[0][1];
        for (int i = 0; i < points.size(); i++) {
            int a = points[i][0], b = points[i][1];
            if (py >= a && py <= b) {
                // cnt++;
            } else {
                py = b;
                cnt++;
            }
        }
        return cnt;
    }
};
```
### 方法1：贪心，按左端点排序
```
class Solution {
public:
    int findMinArrowShots(vector<vector<int>>& points) {
        if (points.size() == 1) return 1;
        sort(points.begin(), points.end());
        int l = points[0][0], r = points[0][1];

        int cnt = 1;

        for (int i = 1; i < points.size(); i++) {
            if (points[i][0] <= r) {
                r = min(r, points[i][1]);
            }
            else {
                cnt++;
                l = points[i][0], r = points[i][1];
            }
        }
        return cnt;
    }
};
```

# 435. 无重叠区间
### 贪心，方法1
```
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),[](vector<int>& a, vector<int>& b) {
            return a[1] < b[1];
        });
        int cnt = 0;
        int l = intervals[0][0], r = intervals[0][1];
        for (int i = 1; i < intervals.size(); i++) {
            int a = intervals[i][0], b = intervals[i][1];
            if (a >= r) {
                l = a, r = b;
                continue;
            } else {
                cnt++;
            }
        }

        return cnt;
    }
};
```
### 贪心，方法2
```
class Solution {
public:
    int eraseOverlapIntervals(vector<vector<int>>& intervals) {
        sort(intervals.begin(), intervals.end(),[](vector<int>& a, vector<int>& b) {
            return a[1] < b[1];
        });

        int cnt = 1;
        int l = intervals[0][0], r = intervals[0][1];
        for (int i = 1; i < intervals.size(); i++) {
            int a = intervals[i][0], b = intervals[i][1];
            if (a >= r) {
                cnt++;
                l = a, r = b;
            } 
        }

        return intervals.size() - cnt;
    }
};
```
# 763. 划分字母区间
### 方法1：贪心 + 哈希表
```
class Solution {
public:
    vector<int> partitionLabels(string s) {
        int n = s.size();
        vector<int> ans;
        unordered_map<char, int> h;
        for (int i = 0; i < n; i++) h[s[i]] = i;
        int l = 0, r = 0;
        for (int i = 0; i < n; i++) {
            r = max(r, h[s[i]]);
            if (r == i) {
                ans.push_back(r - l + 1);
                l = r = i + 1;
            }
        }

        return ans;
    }
};
```
# 738. 单调递增的数字
### 方法1：贪心
```
class Solution {
public:
    int monotoneIncreasingDigits(int n) {
        string s = to_string(n);
        int start = s.size();
        for (int i = s.size() - 1; i; i--) {
            if (s[i - 1] > s[i]) {
                s[i - 1]--;
                
                start = i;
            }
        }
        
        for (int i = start; i < s.size(); i++) {
            s[i] = '9';
        }
        return stoi(s);
    }
};
```
# 714. 买卖股票的最佳时机含手续费
### 方法1：DP
```
class Solution {
public:
    int maxProfit(vector<int>& p, int fee) {
        int n = p.size();
        // dp[i][0] 不持有股票
        // dp[i][1] 持有股票
        vector<vector<int>> dp(n + 1, vector<int>(2, -1E8));
        dp[0][0] = 0;
        // dp[0][1] = 0; 这里不能初始化为0
        int res = 0;
        for (int i = 1; i <= n; i++) {
            dp[i][0] = max(dp[i - 1][0], dp[i - 1][1] + p[i - 1] - fee);
            dp[i][1] = max(dp[i - 1][1], dp[i - 1][0] - p[i - 1]);
            res = max(res, dp[i][0]);
        }

        return res;
    }
};
```
### 方法2：贪心
```
class Solution {
public:
    int maxProfit(vector<int>& prices, int fee) {
        int res = 0;
        int minPrice = prices[0];
        for (int i = 1; i < prices.size(); i++) {
            if (prices[i] < minPrice) {
                minPrice = prices[i];
            }

            if (prices[i] > minPrice + fee) {
                res += prices[i] - minPrice - fee;
                minPrice = prices[i] - fee;
            }
        }

        return res;
    }
};
```
# 968. 监控二叉树
### 方法1：树形DP
```
class Solution {
public:
    int minCameraCover(TreeNode* root) {
        auto dp = dfs(root);

        return min(dp[1], dp[2]);
    }

    // f[0]:当前节点被父节点看
    // f[1]:当前节点被子节点看
    // f[2]:当前节点被自身看
    vector<int> dfs(TreeNode* root) {
        if (!root) return {0, 0, (int)1e8};

        auto l = dfs(root->left), r = dfs(root->right);

        return {
            min(l[1], l[2]) + min(r[1], r[2]),
            min(l[2] + min(r[1],r[2]), r[2] + min(l[1], l[2])),
            min(l[0], min(l[1], l[2])) + min(r[0], min(r[1], r[2])) + 1
        };

    }
};
```

# 746. 使用最小花费爬楼梯
### 方法1:DP
```
class Solution {
public:
    int minCostClimbingStairs(vector<int>& cost) {
        cost.push_back(0);
        int n = cost.size();
        vector<int> dp(n + 1);
        dp[0] = 0;
        dp[1] = 0;
        for (int i = 2; i <= n; i++) {
            dp[i] = min(cost[i - 2] + dp[i - 2], cost[i - 1] + dp[i - 1]);
        }

        return dp[n];
    }
};
```
# 343. 整数拆分
### 方法1：DP
```
class Solution {
public:
    int integerBreak(int n) {
        vector<int> dp(n + 1);
        dp[1] = 1;
        dp[2] = 1;
        for (int i = 3; i <= n; i++) {
            for (int j = 1; j < i; j++) {
                // dp[i] = max(dp[i], max(dp[j] * dp[i - j], j * (i - j)));
                dp[i] = max(dp[i], max(j * dp[i - j], j * (i - j)));
            }
        }

        return dp[n];
    }
};
```

### 方法2：数学 + 贪心
```
class Solution {
public:
    int integerBreak(int n) {
        if (n <= 3) return 1 * (n - 1);
        int p = 1;
        while (n >= 5) {
            n -= 3;
            p *= 3;
        }

        return p * n;
    }
};
```

# 416. 分割等和子集
### 方法1:01背包
```
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int s = accumulate(nums.begin(), nums.end(), 0);
        if (s % 2) return false;
        s /= 2;
        int n = nums.size();
        vector<vector<bool>> dp(n + 1, vector<bool>(s + 1));
        dp[0][0] = true;

        // dp[i][j] 前i个数组组成和为j的可能性
        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= s; j++) {
                dp[i][j] = dp[i][j] || dp[i - 1][j];
                if (j >= nums[i - 1]) {
                    dp[i][j] = dp[i][j] || dp[i - 1][j - nums[i - 1]];
                }
            }
        }

        return dp[n][s];
    }
};
```
### 方法2:01背包空间优化版
```
class Solution {
public:
    bool canPartition(vector<int>& nums) {
        int n = nums.size();
        int sum = accumulate(nums.begin(), nums.end(), 0);
        if (sum % 2) return false;
        sum /= 2;
        vector<bool> dp(sum + 1);

       dp[0] = true;

        for (int i = 1; i <= n; i++) {
            for (int j = sum; j >= nums[i - 1]; j--) {
                dp[j] = dp[j] || dp[j - nums[i - 1]];
            }
        }
        return dp[sum];
    }
};
```

# 1049. 最后一块石头的重量 II
### 方法1：01背包
```
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int s = accumulate(stones.begin(), stones.end(), 0);
        int t = s / 2, n = stones.size();
        vector<vector<int>> dp(n + 1, vector<int>(t + 1));
        dp[0][0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= t; j++) {
                dp[i][j] = max(dp[i][j], dp[i - 1][j]);
                if (j >= stones[i - 1]) {
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - stones[i - 1]] + stones[i - 1]);
                }
            }
        }

        return s - 2 * dp[n][t];
    }
};
```
### 方法2：01背包优化
```
class Solution {
public:
    int lastStoneWeightII(vector<int>& stones) {
        int s = accumulate(stones.begin(), stones.end(), 0);
        int t = s / 2, n = stones.size();

        vector<int> dp(t + 1);
        dp[0] = 0;

        for (int i = 1; i <= n; i++) {
            for (int j = t; j >= 0; j--) {
                if (j >= stones[i - 1]) {
                    dp[j] = max(dp[j], dp[j - stones[i - 1]] + stones[i - 1]);
                }
            }
        }

        return s - 2 * dp[t];
    }
};
```
# 494. 目标和
### 方法1:DFS
```
class Solution {
public:
    int ans;
    int findTargetSumWays(vector<int>& nums, int target) {
        ans = 0;
        dfs(nums, 0, 0, target);

        return ans;
    }

    void dfs(vector<int>& nums, int u, int s, int t) {
        if (u == nums.size()) {
            if (s == t) {
                ans++;
            }
            return;
        }

        for (int i = 0; i < 2; i++) {
            if (i == 0) {
                dfs(nums, u + 1, s + nums[u], t);
            } else if (i == 1) {
                dfs(nums, u + 1, s - nums[u], t);
            }
        }
    }
};
```

### 方法2:DFS
```
class Solution {
public:
    int ans;
    int findTargetSumWays(vector<int>& nums, int target) {
        ans = 0;
        dfs(nums, 0, 0, target);

        return ans;
    }

    void dfs(vector<int>& nums, int u, int s, int t) {
        if (u == nums.size()) {
            if (s == t) {
                ans++;
            }
            return;
        }

        dfs(nums, u + 1, s + nums[u], t);
        dfs(nums, u + 1, s - nums[u], t);
    }
};
```

### 方法3:DP
```
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size();
        int s = accumulate(nums.begin(), nums.end(), 0);
        if (abs(target) > s) return 0;
        if ((target + s) % 2) return 0;
        int x = (target + s) / 2;
        // x - (s - x) = target x为+号之和，则减号之和为s - x
        // 所以 x = (target + s) / 2，转换成装满容量为x的背包有几种方法
        vector<vector<int>> f(n + 1, vector<int>(x + 1));
        f[0][0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 0; j <= x; j++) {
                f[i][j] += f[i - 1][j];
                if (j >= nums[i - 1]) {
                    f[i][j] += f[i - 1][j - nums[i - 1]];
                }
            }
        }


        return f[n][x];
    }
};
```

### 方法4:Dp
```
class Solution {
public:
    int findTargetSumWays(vector<int>& nums, int target) {
        int n = nums.size(), off = 1000;
        vector<vector<int>> f(n + 1, vector<int>(2001, 0));
        f[0][off] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = -1000; j <= 1000; j++) {
                if (j - nums[i - 1] >= -1000)
                    f[i][j + off] += f[i - 1][j - nums[i - 1] + off];
                if (j + nums[i - 1] <= 1000)
                    f[i][j + off] += f[i - 1][j + nums[i - 1] + off];
            }
        }

        return f[n][off + target];
    }
};
```
# 474. 一和零
### 方法1:01背包DP
```
const int N = 610, M = 110;
class Solution {
public:
    int dp[N][M][M];
    int findMaxForm(vector<string>& strs, int m, int n) {
        memset(dp, 0, sizeof dp);
        // dp[i][j][k] 前i个字符串，最多含有j个0, k个1的子集长度
        int len = strs.size();
        for (int i = 1; i <= len; i++) {
            int a = 0, b = 0;
            string t = strs[i - 1];
            for (auto& x : t) {
                if (x == '0') a++;
                else b++;
            }
            // cout << t << " " << a << " " << b << endl;
            for (int j = 0; j <= m; j++) {
                for (int k = 0; k <= n; k++) {
                    // dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k]);
                    dp[i][j][k] = dp[i - 1][j][k];
                    if (j >= a && k >= b)
                        dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j - a][k - b] + 1);
                }
            }
        }

        return dp[len][m][n];
    }
};
```
### 方法2:01背包DP，空间优化
```
const int N = 610, M = 110;
class Solution {
public:
    int dp[M][M];
    int findMaxForm(vector<string>& strs, int m, int n) {
        memset(dp, 0, sizeof dp);
        // dp[j][k] 最多含有j个0, k个1的子集长度
        int len = strs.size();
        for (int i = 1; i <= len; i++) {
            int a = 0, b = 0;
            string t = strs[i - 1];
            for (auto& x : t) {
                if (x == '0') a++;
                else b++;
            }
            // cout << t << " " << a << " " << b << endl;
            for (int j = m; j >= 0; j--) {
                for (int k = n; k >= 0; k--) {
                    // dp[i][j][k] = max(dp[i][j][k], dp[i - 1][j][k]);
                    // dp[i][j][k] = dp[i - 1][j][k];
                    if (j >= a && k >= b)
                        dp[j][k] = max(dp[j][k], dp[j - a][k - b] + 1);
                }
            }
        }

        return dp[m][n];
    }
};
```

# 377. 组合总和 Ⅳ
### 方法1:DP
```
class Solution {
public:
    int combinationSum4(vector<int>& nums, int target) {
        int n = nums.size();
        vector<uint> dp(target + 1);
        dp[0] = 1;

        for (int i = 1; i <= target; i++) {
            for (int j = 0; j < n; j++) {
                if (i >= nums[j]) {
                    dp[i] += dp[i - nums[j]];
                }
            }
        }

        return dp[target];
    }
};
```
### 方法2：记忆化搜索
```
class Solution {
public:
    int res;
    vector<int> cur;      // 当前组合
    int curSum;             // 当前组合的求和
    map<int, int> dict;   // 记忆化结构
    void dfs(vector<int> nums, int tar) {
        if (curSum == tar) {
            res ++;
            return;
        }
        if (curSum > tar) {
            return;
        }
        if (dict.find(tar - curSum) != dict.end()) {
            res += dict[tar - curSum] ;
            return;
        }
        int a = res;
        for (int i = 0; i < nums.size(); i ++) {
            cur.push_back(nums[i]);
            curSum += nums[i];
            dfs(nums, tar);
            curSum -= nums[i];
            cur.pop_back();
        }
        dict[tar - curSum] = res - a;
    }


    int combinationSum4(vector<int>& nums, int target) {
        dfs(nums, target);

        return res;
    }
};
```
#  337. 打家劫舍 III
### 方法1：树形DP
```
class Solution {
public:
    int rob(TreeNode* root) {
        auto v = dfs(root);
        return max(v[0], v[1]);
    }

    vector<int> dfs(TreeNode* root) {
        if (!root) return {0, 0};
        auto l = dfs(root->left);
        auto r = dfs(root->right);
        
        
        return {max(l[0], l[1]) + max(r[0], r[1]), root->val + l[0] + r[0]};
    }
};
```

# LeetCode 188. 买卖股票的最佳时机 IV
### 方法1：soulmachine的做法，二维DP
```
class Solution {
public:
    int maxProfit(int k, vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> sell(n, vector<int>(k + 1));
        vector<vector<int>> buy(n, vector<int>(k + 1));
        buy[0][0] = -prices[0];
        sell[0][0] = 0;
        for (int i = 1; i <= k; i++) sell[0][i] = buy[0][i] = INT_MIN / 2;
        int res = 0;

        for (int i = 1; i < n; i++) {
            buy[i][0] = max(buy[i - 1][0], sell[i - 1][0] - prices[i]);
            res = max(res, buy[i][0]);
            for (int j = 1; j <= k; j++) {
                buy[i][j] = max(buy[i - 1][j], sell[i - 1][j] - prices[i]);
                sell[i][j] = max(sell[i - 1][j], buy[i - 1][j - 1] + prices[i]);
                res = max(res, sell[i][j]);
            }
            // res = max(res, sell[i][k]);
        } 
        // cout << res << endl;
        // return *max_element(sell[n - 1].begin(), sell[n - 1].end());
        return res;
    }
};
```

### 方法2：soulmachine的做法，优化空间一维DP
```
class Solution {
public:
    int maxProfit(int k, vector<int>& p) {
        int n = p.size();
        vector<int> sell(k + 1);
        vector<int> buy(k + 1, INT_MIN);
        
        for (int i = 0; i < n; i++) {
            for (int j = 1; j <= k; j++) {
                buy[j] = max(buy[j], sell[j - 1] - p[i]);
                sell[j] = max(sell[j], buy[j] + p[i]);
            }
        }

        return sell[k];
    }
};
```

# LeetCode 309. 最佳买卖股票时机含冷冻期
### 方法1：自己的做法
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        vector<vector<int>> f(n + 1, vector<int>(3));
        // f[i][0] 买入, 有股票
        // f[i][1] 卖出, 无股票
        // f[i][2] 冷冻期， 无股票

        for (int i = 1; i <= n; i++) {
            if (i > 1) {
                f[i][0] = max(f[i - 1][0], f[i - 2][1] - prices[i - 1]);
                f[i][1] = max(f[i - 1][0] + prices[i - 1], max(f[i - 1][1], f[i - 1][2]));
                f[i][2] = f[i - 1][1];
            } else {
                f[i][0] = - prices[i - 1];
                f[i][1] = 0;
                f[i][2] = 0;
            }
        }  

        return max(f[n][0], max(f[n][1], f[n][2]));   
    }
};
```

### 方法2：DP
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        if (prices.empty()) return 0;
        int n = prices.size();
        vector<vector<int>> f(n, vector<int>(3, -1e8));
        f[0][0] = 0; // f[i][0] 进入冷冻期
        f[0][1] = -prices[0]; // f[i][1] 已买入 f[i][2]今天卖出

        for (int i = 1; i < n; i++) {
            f[i][0] = max(f[i - 1][0], f[i - 1][2]);
            f[i][1] = max(f[i - 1][1], f[i - 1][0]- prices[i]);
            f[i][2] = f[i - 1][1] + prices[i];
        }

        return max(f[n - 1][0], max(f[n - 1][1], f[n - 1][2]));
    }
};
```

### 方法3：DP
```
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n = prices.size();
        if (n == 0)
            return 0;

        vector<int> f(n); // 当天不持有的收益
        vector<int> g(n); // 当天持有的收益

        f[0] = 0;
        g[0] = -prices[0];
        for (int i = 1; i < n; i++) {
            f[i] = max(f[i - 1], g[i - 1] + prices[i]);
            if (i >= 2)
                g[i] = max(g[i - 1], f[i - 2] - prices[i]);
            else
                g[i] = max(g[i - 1], -prices[i]);
        }
        return f[n - 1];
    }
};
```
# 1035. 不相交的线
### 方法1:Dp
```
class Solution {
public:
    int maxUncrossedLines(vector<int>& nums1, vector<int>& nums2) {
        int n = nums1.size(), m = nums2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (nums1[i - 1] == nums2[j - 1]) {
                    dp[i][j] = max(dp[i][j], dp[i - 1][j - 1] + 1);
                } else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[n][m];
    }
};
```
# 392. 判断子序列
### 方法1：双指针
```
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int n = s.size(), m = t.size();
        int i = 0, j = 0;
        while (j < m && i < n) {
            if (s[i] == t[j]) {
                i++;
                j++;
            } else {
                j++;
            }
        }

        return i == n;
    }
};
```
### 方法2:Dp
```
class Solution {
public:
    bool isSubsequence(string s, string t) {
        int n = s.size(), m = t.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        dp[0][0] = 0;
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s[i - 1] == t[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                }  else {
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1]);
                }
            }
        }

        return dp[n][m] == n;
    }
};
```
# 115. 不同的子序列
### 方法1:Dp
```
class Solution {
public:
    int numDistinct(string s, string t) {
        int n = s.size(), m = t.size();
        vector<vector<uint>> dp(n + 1, vector<uint>(m + 1));
        for (int i = 0; i <= n; i++) dp[i][0] = 1;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (s[i - 1] == t[j - 1]) {
                    dp[i][j] = dp[i - 1][j] + dp[i - 1][j - 1];
                } else {
                    dp[i][j] = dp[i - 1][j];
                }
            }
        }

        return dp[n][m];
    }
};
```
# P151 583. 两个字符串的删除操作
### 方法1：最长子序列的变形 DP
```
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> dp(n + 1, vector<int>(m + 1));
        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                if (word1[i - 1] == word2[j - 1]) {
                    dp[i][j] = dp[i - 1][j - 1] + 1;
                } else {
                    dp[i][j] = max(dp[i][j - 1], dp[i - 1][j]);
                }
            }
        }
        int len = dp[n][m];
        return n + m - len * 2;
    }
};
```
### 方法2：DP
```
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) f[i][0] = i;
        for (int i = 1; i <= m; i++) f[0][i] = i;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
                if (word1[i - 1] == word2[j - 1]) {
                    f[i][j] = min(f[i][j], f[i - 1][j - 1]);
                }
            }
        }
        return f[n][m];
    }
};
```
### 方法3：DP
```
class Solution {
public:
    int minDistance(string word1, string word2) {
        int n = word1.size(), m = word2.size();
        vector<vector<int>> f(n + 1, vector<int>(m + 1, 0));
        for (int i = 1; i <= n; i++) f[i][0] = i;
        for (int i = 1; i <= m; i++) f[0][i] = i;

        for (int i = 1; i <= n; i++) {
            for (int j = 1; j <= m; j++) {
                
                if (word1[i - 1] == word2[j - 1]) {
                    // f[i][j] = min(f[i][j], f[i - 1][j - 1]);
                    f[i][j] = f[i - 1][j - 1];
                } else {
                    f[i][j] = min(f[i - 1][j], f[i][j - 1]) + 1;
                }
            }
        }
        return f[n][m];
    }
};
```
# 647. 回文子串
### 方法1：迭代做法
```
class Solution {
public:
    int countSubstrings(string s) {
        int cnt = 0;
        int n = s.size();
        for (int i = 0; i < n; i++) {
            int l = i, r = i;
            while (l >= 0 && r < n) {
                if (s[l] == s[r]) {
                    cnt++;
                    l--;
                    r++;
                } else break;
            }
            l = i, r = i + 1;
            while (l >= 0 && r < n) {
                if (s[l] == s[r]) {
                    cnt++;
                    l--;
                    r++;
                } else break;
            }
        }

        return cnt;
    }
};
```
