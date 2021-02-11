---
title: WinterTrainingDay1
date: 2021-02-01 19:59:21
tags: [code, jxyz, training]
mathjax: true
---

# 高一寒假集训 Day1 记

<!-- more -->

&emsp;&emsp;今天是紧张而刺激的一天！

&emsp;&emsp;早晨如愿早起吃早饭，然后提前一个小时来到机房。于是有了这个博客（

&emsp;&emsp;上午学习的是线性 DP 的两个基础模型：LIS（最长上升子序列）和 LCS（最长公共子序列）。

## 最长上升子序列问题（LIS）

&emsp;&emsp;给定一个序列 $a_n$ ，输出最大单调递增序列的长度。定义 $f(i)$ 为**到达 $i$ 位置的最长上升子序列长度**，枚举 $j$ ，满足 $j<i$ 且 $a_j<a_i$ ，那么 $a_i$ 就可以接在 $a_j$ 的后面，构成一个更长的上升子序列，其长度为 $f(j)+1$ 。**注意初始值：在每个位置没更新前，$f$ 值就是 $1$ 。**

&emsp;&emsp;代码如下：

```c++
for(int i = 1; i <= n; ++i) {
    f[i] = 1; //注意初始值的更新！
    for(int j = 1; j < i; ++j) { //注意枚举的范围是1~i-1！
        if(a[j] < a[i]) f[i] = max(f[i], f[j]+1); //是当前长度（f[i]）长呢，还是更新后的长度（f[j]+1）长，注意比较
    }
}
for(int i = 1; i <= n; ++i) {
    ans = max(f[i], ans); //ans的初始值为-INF。
}
```

&emsp;&emsp;也可以更换对 $f$ 的定义，让 $f(x)$ 表示**以数字 $x$ 结尾的最长上升子序列长度**。依然从 $1$ 到 $n$ 枚举序列的每个位置，对于每一个位置 $i$ 有 
$$
f(a_i) = min\{f(j), 1\leqslant j\leqslant a_i\}.
$$
&emsp;&emsp;重要的是，应该按照数字在原序列中的位置进行枚举。这个做法可以使用**树状数组**进行优化。

## 最长公共子序列问题（LCS）

&emsp;&emsp;求两个序列的最长公共子序列的长度。设 $f(i,j)$ 为 $A_1,A_2,\cdots ,A_i$ 和 $B_1,B_2,\cdots ,B_i$ 的 LCS 长度，则
$$
f(i,j)=\begin{cases}
max\{f(i,j),f(i-1,j-1)+1\} & A[i]=A[j] \\
max\{f(i-1,j),f(i,j-1)\} & else
\end{cases}.
$$
&emsp;&emsp;代码如下：

```c++
for(int i = 1; i <= n; ++i) {
	for(int j = 1; j <= m; ++j) {
		f[i][j] = max(f[i-1][j], f[i][j-1]);
		if(a[i] == b[j])
			f[i][j] = max(f[i][j], f[i-1][j-1]+1);
	}
}
ans = f[n][m];
```

&emsp;&emsp;考虑另一种做法，将第二个串中的每个元素倒序展开为第一个串对应元素出现的位置，这样选出一个上升子序列，就对应选出一个上下两个串的公共子序列。要让公共子序列最长，就需要让上升子序列最长。这就将 LCS 问题转化为了 LIS 问题。

&emsp;&emsp;其中，倒序是为了**让后面具有相同大小的元素排在前面，保证 LCS 的长度**。

## 练习

### 导弹拦截（洛谷 [P1020](https://www.luogu.com.cn/problem/P1020)）

&emsp;&emsp;目前的水平只能过前一半，另一半需要树状数组来解决……不过还是写一下做题心得吧。

&emsp;&emsp;因为是 “都不能高于” ，所以 Q1 求的是最长不上升子序列的长度，模板套一下就好；

&emsp;&emsp;而对于 Q2 要求拦截所有导弹时最少配备的系统数，~~（看了看题解，）~~可证明解为最长上升子序列。不会就不说了吧（

> Dilworth 定理：偏序集的最少反链划分数等于最长链的长度。

&emsp;&emsp;然后是写题过程中遇到的一些注意点：1. 对于 DP 的题目，应该先在草稿纸上列出状态转移方程，设计一个比较好的状态；2. **注意初始值**。

&emsp;&emsp;对于这种没有限定输入数据的，可以使用以下方式解决。

```c++
while(cin >> H[n++]); // ++n 或者 n++ 取决于初始值和后面的设定。
```

&emsp;&emsp;最终，100 分（满分 200）代码如下：

```c++
//Luogu P1020 [NOIP1999 普及组] 导弹拦截
#include<iostream>
#include<cstring>
#include<cstdio>
using namespace std;

const int MAXN = 1e5+10;
int H[MAXN], f[MAXN];
int ans1, ans2;

int main () {
	while(cin >> H[n++]);
	for(int i = 1; i < n; ++i) {
		f[i] = 1;
		for(int j = 1; j < i; ++j) {
			if(H[j] >= H[i]) f[i] = max(f[i], f[j]+1);
		}
	}
	for(int i = 1; i < n; ++i) ans1 = max(ans1, f[i]);
	memset(f,0,sizeof(f));
	for(int i = 1; i < n; ++i) {
		f[i] = 1;
		for(int j = 1; j < i; ++j) {
			if(H[j] < H[i]) f[i] = max(f[i], f[j]+1);
		}
	}
	for(int i = 1; i < n; ++i) ans2 = max(ans2, f[i]);
	printf("%d\n%d", ans1, ans2);
	return 0;
} 

```

&emsp;&emsp;~~（不是满分的代码还拿出来）~~

&emsp;&emsp;~~（有时间学习树状数组吧）~~

## 01 背包

### 概述

&emsp;&emsp;有 $N$ 件物品和一个容量为 $V$ 的背包。放入第 $i$ 件物品耗费的容量是 $v_i$ ，得到的价值是 $w_i$ 。**每种物品仅有一件**，可以选择放或不放。求解将哪些物品装入背包可使得价值总和最大。

&emsp;&emsp;用子问题定义状态，则 $f_{i,j}$ 表示前 $i$ 个物品放入容量为 $j$ 的背包的最大价值，则其状态转移方程便是：
$$
f_{i,j} = max\{f_{i-1,j},f_{i-1,j-v_i} + w_i\}.
$$
&emsp;&emsp;这个方程是所有背包问题的基础。 “前 $i$ 个物品放入容量为 $j$ 的背包” 这个子问题，若只考虑第 $i$ 件物品的策略（放或不放），那么就可以转化为一个只和前 $i-1$ 件物品相关的问题。如果不放第 $i$ 件物品，那么问题就转化为 “前 $i-1$ 件物品放入容量为 $j$ 的背包中” ，价值为 $f_{i-1, j}$ ；如果放第 i 件物品，那么问题就转化为 “前 $i-1$ 件物品放入剩下的容量为 $j-v_i$ 的背包中” ，价值为 $f_{i-1,j-v_i}+w_i$ 。

&emsp;&emsp;代码如下：

```c++
memset(dp, 0, sizeof dp); //用 memset() 一定要记得加 cstring 库！
for(int i = 1; i <= N; ++i) {
	for(int j = v[i]; j <= V; ++j) {
		f[i][j] = max(f[i-1][j], f[i-1][j-v[i]] + w[i]);
	}
}
```

### 初始化

&emsp;&emsp;在**所有求最优解的背包问题**中，有两种不大相同的问法。

&emsp;&emsp;第一种问法要求**恰好装满背包**。那么在初始化时除了 $f_0 \leftarrow 0$ ，其它 $f_{1,2, \cdots ,V}$ 均设为 $f_{1,2, \cdots ,V} \leftarrow -\infty$ ，以保证最终得到的 $f_V$ 是一种恰好装满背包的最优解。

&emsp;&emsp;第二种问法**没有要求必须把背包装满**，只是希望价值尽量大，那么在初始化时应该将 $f_{0,1, \cdots,V}$ 均设为 $f_{0,1, \cdots, V} \leftarrow 0$ 。这是因为初始化的 $f_{0,1, \cdots,V}$ 事实上就是在没有任何物品可以放入背包时的状态。如果要求**恰好装满背包**，那么 $f_{1,2, \cdots ,V}$ 在未装入任何东西的时候没有合法的解。如果**背包并非必须装满**，那么 $f_{1,2, \cdots ,V}$ 在未装入任何东西的时候的解都是合法的。

&emsp;&emsp;这个小技巧完全可以推广到其他类型的背包问题。

### 优化

#### 空间复杂度优化

&emsp;&emsp;以上方法的时间和空间复杂度均为 $O(VN)$ ，空间复杂度可以优化到 $O(V)$ 。这要求在每次主循环中我们以 $j \leftarrow V,V-1,\cdots,v[i]$ 的**递减顺序**计算 $f_j$ ，以保证计算 $f_j$ 时 $f_{j-v_i}$ 保存的是状态 $f_{i-1, j-v_i}$ 的值。代码如下：

```c++
memset(dp, 0, sizeof dp);
for(int i = 1; i <= N; ++i) {
    for(int j = V; j >= v[i]; --j) {
        dp[j] = max(dp[j], dp[j-v[i]] + w[i]);
    }
}
```

##### 小技巧

&emsp;&emsp;我们可以将 `f[j] = max(f[j], f[j-v[i]] + w[i])` 作为一个函数 `gmax()` ，简化代码：

```c++
inline void gmax(int &x, const int &y) { //const + 传值引用& 加快速度
    if(x < y) x = y;
}
```

#### 常数优化

&emsp;&emsp;还可以对代码中的第二重循环的下限进行常数优化：当 V 大到可以把从第 i 个到后面所有的物品全部装入背包而且剩余空间仍大于第 i 个物品的空间时，可以减少不必要的循环，不需要更新到过左的位置，即 $j \leftarrow V,V-1, \cdots ,V-\sum_{k=i+1}^Nv_k$ 。求 $\sum_{k=i+1}^Nv_k$ **可以用前缀和**。

&emsp;&emsp;代码如下：

```c++
memset(dp, 0, sizeof dp);
for(int i = 1; i <= N; ++i) s[i] = s[i-1] + w[i];
for(int i = 1; i <= N; ++i) {
    int bound = max(V - (s[N] - s[i]), w[i]);
	for(int j = V; j >= bound; --j) {
		gmax(dp[j-v[i]] + w[i], dp[j]);
	}
}
```

## 完全背包

### 概述

&emsp;&emsp;有 $N$ 件物品和一个容量为 $V$ 的背包。放入第 $i$ 件物品耗费的容量是 $v_i$ ，得到的价值是 $w_i$ 。**每种物品都有无限件可用**，可以选择放或不放。求解将哪些物品装入背包可使得价值总和最大，且这些物品耗费的容量总和不超过背包容量。

&emsp;&emsp;从每种物品的角度考虑，与它相关的策略有取 0 件、取 1 件、取 2 件、$\cdots$、取 $\lfloor \frac{V}{v_i} \rfloor$ 件。如果仍按照解 01 背包问题的思路，令 $f_{i,j}$ 表示前 $i$ 种物品恰放入一个容量为 $j$ 的背包的最大价值，仍然可以写出状态转移方程：
$$
f_{i,j} = max\{f_{i-1, j-k\times v_i}+k\times w_i | 0\leqslant k\times v_i\leqslant j\}
$$
&emsp;&emsp;求解状态 $f_{i,j}$ 的时间是 $O(\frac{j}{v_i})$ ，总的复杂度可以认为是 $O(VN\sum\frac{V}{v_i})$ ，是比较大的。

### 优化

#### 一些有效的优化

&emsp;&emsp;若两件物品 $i$ 、 $j$ 满足 $v_i \leqslant v_j$ 且 $w_i \geqslant w_j$ ，那么可以将物品 $j$ 直接去掉，不用考虑。因为任何情况下都可以将价值小费用高的 $j$ 换成物美价廉的 $i$ ，得到的方案至少不会更差。但是这并不能改善最坏情况的复杂度，因为有可能特别设计的数据可以一件物品也去不掉。

&emsp;&emsp;考虑到第 $i$ 种物品最多可以选 $\lfloor \frac{V}{v_i}\rfloor$ 件，于是可以把第 $i$ 种物品转化为 $\lfloor \frac{V}{v_i}\rfloor$ 件费用及价值均不变的物品，然后求解这个 01 背包问题。

&emsp;&emsp;更高效的转化方法是考虑**二进制**的思想：把第 $i$ 种物品拆成费用为 $v_i\times 2^k$ 、价值为 $w_i\times 2^k$ 的若干件物品，其中 $k$ 取满足 $v_i\times 2^k\leqslant V$ 的非负整数。不管最优策略选几件第 $i$ 种物品，其件数写成二进制后，总可以表示成若干个 $2^k$ 件物品的和。这样一来就把每种物品拆成 $O(\log_2 \lfloor \frac{V}{v_i}\rfloor)$ 件物品。但是我们有更优的 $O(VN)$ 时间复杂度的算法。

#### $O(VN)$ 的算法

&emsp;&emsp;这个算法使用一维数组。代码如下：

```c++
memset(dp, 0, sizeof dp);
for(int i = 1; i <= N; ++i) {
    for(int j = v[i]; j <= V; ++j) {
        gmax(dp[j], dp[j-v[i]] + w[i]);
    }
}
```

&emsp;&emsp;在 01 背包中要按照 $j$ 递减的次序来循环，是为了保证第 $i$ 次循环中的状态 $f_{i,j}$ 是由状态 $f_{i-1, j-v_i}$ 递推而来。也就是为了保证每件物品只选一次，使得在考虑 “选入第 $i$ 件物品” 这件策略时，依据的是一个绝无已经选入第 $i$ 件物品的子结果 $f_{i-1, j-v_i}$ 。而现在完全背包的特点恰是每种物品可以选择无限件，所以在考虑 “加选一件第 $i$ 种物品” 这种策略时，需要一个可能已选入第 $i$ 种物品的子结果 $f_{i,j-v_i}$ ，所以就必须采用 $j$ **递增**的顺序循环。

## 多重背包

### 概述

&emsp;&emsp;有 $N$ 种物品和一个容量为 $V$ 的背包。第 $i$ 种物品最多有 $M_i$ 件可用，每件耗费的空间是 $v_i$ ，价值是 $w_i$ 。求解将哪些物品装入背包可以使得这些物品的耗费空间总和不超过背包容量，且价值总和最大。对于第 $i$ 种物品有 $M_i+1$ 种策略：取 0 件、取 1 件、$\cdots$ 、取 $M_i$ 件。令 $f_{i,j}$ 表示前 $i$ 种物品恰好放入一个容量为 $j$ 的背包的最大价值，则有状态转移方程：
$$
f_{i,j} = max\{f_{i-1,j-k*v_i}+k*w_i|0\leqslant k\leqslant M_i\}.
$$
&emsp;&emsp;时间复杂度为 $O(V\sum M_i)$ 。

### 优化

&emsp;&emsp;转化为 01 背包问题求解：把第 $i$ 种物品换成 $M_i$ 件 01 背包中的物品，则转化为物品数为 $\sum M_i$ 的 01 背包问题。时间复杂度仍为 $O(V\sum M_i)$ 。

&emsp;&emsp;仍然考虑**二进制**的思想，把第 $i$ 种物品换成若干件物品，使得原问题中第 $i$ 种物品可取的每种策略 —— 取 0 件、取 1 件、$\cdots$ 、取 $M_i$ 件 —— 均能等价于取若干件代换以后的物品。另外，取超过 $M_i$ 件的策略必不能出现。方法是：将第 $i$ 种物品分成若干件 01 背包中的物品，其中每件物品有一个系数。这件物品的费用和价值均是原来的费用和价值乘以这个系数。令这些系数分别为 $1,2,2^2,\cdots ,2^{k-1},M_i-2^k+1$ ，且 $k$ 为满足 $k < \log (M_i+1)$ 的最大整数。例如，若 $M_i$ 为 13 ，则相应的 $k$ = 3 ，这种最多取 13 件的物品应被分成系数分别为 $1，2，4，6$ 的四件物品。（ $13-2^3+1=6.$ ）分成的这几件物品的系数和为 $M_i$ ，表明不可能取多于 $M_i$ 件的第 $i$ 种物品。这样就将第 $i$ 种物品分成了 $O(\log M_i)$ 种物品，将原问题转化为了时间复杂度为 $O(V\sum\log M_i)$ 的 01 背包问题，是很大的改进。代码如下：

```c++
inline void gmin(int &a, const int &b) {
    if(a > b) a = b;
}
int number[A];
memset(dp, 0, sizeof dp);
for(int i = 1; i <= n; ++i) {
    int num = min(number[i], V/v[i]);
    for(int k = 1; num > 0; k <<= 1) {
        gmin(k, num);
        num -= k;
        for(int j = V; j >= v[i]*k; --j) 
            gmax(dp[j], dp[j-v[i]*k] + w[i]*k);
    }
}
ans  = dp[V];
```