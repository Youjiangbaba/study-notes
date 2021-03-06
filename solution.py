import numpy as np

class Solution:
    '''
        121. 股票最大收益
    '''
    def maxProfit(self, prices):
        sumProfit = 0
        flagStart = 0
        sellIndex,buyIndex = 0,0
        prices.append(-1)
        for i in range(len(prices) -1):
            if prices[i] > prices[i+1]:
                if flagStart == 1:
                    sellIndex = i
                    mid = (prices[sellIndex] - prices[buyIndex])
                    sumProfit += mid
                    flagStart = 0
            else:
                if flagStart == 0:
                    buyIndex = i
                    flagStart = 1
            # print(sumProfit,buyIndex,sellIndex)
        return sumProfit
    
    '''
        1. 两数之和
    '''
    def twoSum(self, nums, target):
        # dictKey = [i for i in range(len(nums))]
        # dictNums = dict(zip(dictKey,nums))
        dictNums = {}

        for i,num1 in enumerate(nums):
            num2 = target - num1
            if num2 in dictNums:
                return [dictNums[num2],i]
            dictNums[num1] = i
        return None
    
    '''
        973. 最接近原点的 K 个点
    '''
    def kClosest(self, points, K):
        xy = lambda a:a[0]**2 + a[1]**2
        points.sort(key = xy)
        return points[:K]

    '''
        242. 有效的字母异位词
    '''
    def isAnagram(self, s, t):
        if len(s) != len(t):
            return False
        import numpy as np
        indexArr = [0 for i in  range(26)]
        for i in range(len(s)):
            index1 = ord(s[i]) - ord('a')
            index2 = ord(t[i]) - ord('a')
            indexArr[index1] += 1
            indexArr[index2] -= 1
        if np.sum(np.abs(indexArr)) == 0:
            return True
        else:
            return False

        if 1:#hash表
            count = {}
            for char in s:
                if char in count:
                    count[char] += 1
                else:
                    count[char] = 1
            for char in t:
                if char in count:
                    count[char] -= 1
                else:
                    return False
            for value in count.values():
                if value != 0:
                    return False
            return True

    '''
        31. 下一个排列
            实现获取下一个排列的函数，算法需要将给定数字序列重新排列成字典序中下一个更大的排列。
            如果不存在下一个更大的排列，则将数字重新排列成最小的排列（即升序排列）。
            必须原地修改，只允许使用额外常数空间。

            以下是一些例子，输入位于左侧列，其相应输出位于右侧列。
            1,2,3 → 1,3,2
            3,2,1 → 1,2,3
            1,1,5 → 1,5,1

            字典序：1 2 3 | 1 3 2 | 2 1 3 | 2 3 1 | 3 1 2 | 3 2 1
    '''
    def nextPermutation(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        """
        print(nums)
        if len(nums) == 1:
            return 

        dealIndex = 0
        dealNum = 0
        for i in range(1,len(nums)):
            if nums[-i] > nums[-i -1]:#从右到左，左边小于右边
                #记录这个左边的数与索引
                dealIndex = -i -1 + len(nums)
                dealNum = nums[dealIndex]
                i = -100
                break
        
        if i != -100:#表示没有找到
            nums.reverse()
        else:
            #标记后面的数，从小到大排序
            b = nums[dealIndex+1:]
            b.sort(reverse=False)
            nums[dealIndex+1:] = b
            for  i in range(dealIndex+1,len(nums)):#找大于标记值的最小数，进行替换
                if nums[i] > nums[dealIndex]:
                    mid = nums[dealIndex]
                    nums[dealIndex] = nums[i]
                    nums[i] = mid
                    break
        print(nums)

    '''
    1122. 数组的相对排序

        给你两个数组，arr1 和 arr2，
            arr2 中的元素各不相同
            arr2 中的每个元素都出现在 arr1 中
        对 arr1 中的元素进行排序，使 arr1 中项的相对顺序和 arr2 中的相对顺序相同。未在 arr2 中出现过的元素需要按照升序放在 arr1 的末尾。

        示例：
        输入：arr1 = [2,3,1,3,2,4,6,7,9,2,19], arr2 = [2,1,4,3,9,6]
        输出：[2,2,2,1,4,3,3,9,6,7,19]
    '''
    def relativeSortArray(self, arr1, arr2):
        
        dict2 = dict(zip(arr2,[0 for i in arr2]))
        #内存消耗后百分五
        # arr3 = []
        # for a1 in arr1:
        #     # a in dict 相当于  a in dict.keys()   ;而 d1.values()为值
        #     if a1 in dict2:
        #         dict2[a1] += 1
        #     else:
        #         arr3.append(a1)
        def get_same(arr1,i):
            if i == len(arr1):
                return arr1
            else:
                if arr1[i] in dict2:
                    dict2[arr1[i]] += 1
                    del arr1[i]
                    return get_same(arr1,i)
                else:
                    return get_same(arr1,i+1)
        arr1 = get_same(arr1,0)
        results = []
        for key in dict2.keys():
            results = results + [key for i in range(dict2[key])]
        
        #排序
        arr1.sort()
        print(results,arr1)
        return results+arr1

    '''
    1030. 距离顺序排列矩阵单元格
        给出 R 行 C 列的矩阵，其中的单元格的整数坐标为 (r, c)，满足 0 <= r < R 且 0 <= c < C。
        另外，我们在该矩阵中给出了一个坐标为 (r0, c0) 的单元格。
        返回矩阵中的所有单元格的坐标，并按到 (r0, c0) 的距离从最小到最大的顺序排，其中，两单元格(r1, c1) 和 (r2, c2) 之间的距离是曼哈顿距离，|r1 - r2| + |c1 - c2|。
        使用桶排序，根据曼哈顿距离，将坐标放到对应的桶中。具体的思路如下：
            首先先求得坐标之间的最大曼哈顿距离；（见示例，坐标之间的距离列表中，可能存在相同的距离，但会有一个最大的距离。）
            根据求得的最大曼哈顿距离，确定桶的数量。根据曼哈顿距离分桶，相同的放到同个桶中；
            最后，将桶中的坐标，根据距离大小添加到结果列表中。
    '''
    def allCellsDistOrder(self, R, C, r0, c0):
        def get_d(r, c, r0, c0):
            """求曼哈顿距离
            """
            return abs(r-r0) + abs(c-c0)
        
        # 确定曼哈顿距离的最大值，进而确定桶数量
        max_d = max(r0, R - 1 - r0) + max(c0, C - 1 - c0)
        # 初始化桶，遍历将坐标放入根据曼哈顿距离放到对应的桶中
        bucket = {}
        for r in range(R):
            for c in range(C):
                d = get_d(r, c, r0, c0)
                if d not in bucket:
                    bucket[d] = []
                bucket[d].append([r, c])
        
        # 将桶中元素添加到结果列表中
        res = []
        for i in range(max_d+1):
            res.extend(bucket[i])
        return res

    '''
    402. 移掉K位数字
        给定一个以字符串表示的非负整数 num，移除这个数中的 k 位数字，使得剩下的数字最小。
        注意:
            num 的长度小于 10002 且 ≥ k。
            num 不会包含任何前导零。
        示例 1 :
            输入: num = "1432219", k = 3
            输出: "1219"
            解释: 移除掉三个数字 4, 3, 和 2 形成一个新的最小的数字 1219。
    '''
    def  removeKdigits(self, num, k):
        #每次扣除一个，比较字典序，得到最小字典序；再进行扣除 ——————超时了
        if 0:
            if k == len(num):
                return '0'
            result = ''
            min_str = ''
            min_num = int(num)
            min_index = 0
            for i in range(k):
                for j in range(len(num)):
                    num1 = num[:j] + num[j+1 :]
                    if int(num1) <= min_num:
                        min_str = num1
                        min_num = int(num1)
                        min_index = j
                result += num[min_index]
                num = min_str
                # print(min_index,num[min_index],num)
            print(result,min_str)
            return str(min_num)
        
        #从左到右遍历，如果这个数大于后一个，应该丢弃——————这种会遇到 1230 3 这种判断失败
        #从右到左遍历，如果这个数小于后一个，丢弃后一个 ————这样会导致，多扣除数，剩下的就为更大的了  1432219 3
        if 0:
            def drop(num,k,i):
                print(num)
                if i == len(num) or k == 0:
                    return num
                else:
                    if num[len(num)-i] < num[len(num)- i - 1]:
                        num = num[:len(num)- i - 1] + num[len(num)-i:]
                        k -= 1
                        return drop(num,k,i)
                    else:
                        return drop(num,k,i + 1)

            #从左到右遍历，如果这个数大于后一个，应该丢弃
            def drop2(num,k,i):
                if i == len(num) - 1 or k == 0:
                    return num
                else:
                    if num[i] > num[i + 1]:
                        num = num[:i] + num[i+1:]
                        k -= 1
                        return drop2(num,k,i)
                    else:
                        return drop2(num,k,i + 1)
            if k == len(num):
                return '0'
            result = drop2(num,k,0)
            new_k = k - (len(num) - len(result))

            while new_k > 0: #在 112 1内，会new_k = 1死循环
                result = drop2(result,new_k,0)
                new_k = k - (len(num) - len(result))
                print(new_k)
            print(str(int(result)))
            return str(int(result))
        '''
            使用栈这种在一端进行添加和删除的数据结构
            初始化空栈，每次向栈中添加一个元素；在添加下一个前，先判断：栈顶（上一次添加的）是否大于这个数，如果大于，那么栈顶的数就是不需要的了,k--
        '''
        stack = []
        remain = len(num) - k
        for digit in num:
            while k and stack and stack[-1] > digit:
                n = stack.pop()#stack.pop()和stack.peek()的区别,都返回栈顶的值，pop删除
                k -= 1
                print("pop:%s"%n,stack,k)
            stack.append(digit)
            print(stack)
        result = "".join(stack[:remain])#''.join(stack[:remain]).lstrip('0') or '0'
        result = str(int(result))
        print(result)
        return result
        

    '''
    134. 加油站
        在一条环路上有 N 个加油站，其中第 i 个加油站有汽油 gas[i] 升。
        你有一辆油箱容量无限的的汽车，从第 i 个加油站开往第 i+1 个加油站需要消耗汽油 cost[i] 升。你从其中的一个加油站出发，开始时油箱为空。
        如果你可以绕环路行驶一周，则返回出发时加油站的编号，否则返回 -1。
        说明: 

            如果题目有解，该答案即为唯一答案。
            输入数组均为非空数组，且长度相同。
            输入数组中的元素均为非负数。
        示例 1:
            输入: 
            gas  = [1,2,3,4,5]
            cost = [3,4,5,1,2]
            输出: 3
            解释:
            从 3 号加油站(索引为 3 处)出发，可获得 4 升汽油。此时油箱有 = 0 + 4 = 4 升汽油
            开往 4 号加油站，此时油箱有 4 - 1 + 5 = 8 升汽油
            开往 0 号加油站，此时油箱有 8 - 2 + 1 = 7 升汽油
            开往 1 号加油站，此时油箱有 7 - 3 + 2 = 6 升汽油
            开往 2 号加油站，此时油箱有 6 - 4 + 3 = 5 升汽油
            开往 3 号加油站，你需要消耗 5 升汽油，正好足够你返回到 3 号加油站。
            因此，3 可为起始索引。
    '''
    def canCompleteCircuit(self, gas, cost):
        '''
            分析：需要完成满足两个条件。1、从i能开到i+1； 2、第i时，总油量大于0;
            假设从编号为0站开始，一直到k站都正常，在开往k+1站时车子没油了。这时，应该将起点设置为k+1站。
            因为k->k+1站耗油太大，0->k站剩余油量都是不为负的，每减少一站，就少了一些剩余油量。所以如果从k前面的站点作为起始站，剩余油量不可能冲过k+1站。
            从k+1站点出发可以开完全程？因为，起始点将当前路径分为A、B两部分。其中，必然有(1)A部分剩余油量<0。(2)B部分剩余油量>0。所以，无论多少个站，都可以抽象为两个站点（A、B）。
            (1)从B站加满油出发，(2)开往A站，车加油，(3)再开回B站的过程。重点：B剩余的油>=A缺少的总油。必然可以推出，B剩余的油>=A站点的每个子站点缺少的油
        '''
        run = 0
        rest = 0
        start = 0
        for i in range(len(gas)):
            run += (gas[i] - cost[i])
            rest += (gas[i] - cost[i])
            if run < 0:
                start = i + 1
                run = 0
            print("run:%d, res:%d ,i:%d"%(run,rest,i))
        if rest < 0:
            return -1
        else:
            return start
             
            
    '''
        283. 移动零
        给定一个数组 nums，编写一个函数将所有 0 移动到数组的末尾，同时保持非零元素的相对顺序。(只能在nums操作)
    '''
    def moveZeroes(self, nums):
        """
        Do not return anything, modify nums in-place instead.
        1\递归判断某元素，如果为0，就删除，且list尾部append 0，且moved+=1；结束为i大于len-moved
        2\双指针，一次遍历   如果不等于0，就把该数放在相对左边；等于01，就相对右边；遍历一次也就把0都移到了右边
        """
        def move(nums,i,moved):
            # print(nums,i)
            if i >= len(nums) - moved:
                return nums
            if nums[i] == 0:
                del nums[i]
                nums.append(0)
                moved += 1
                return move(nums,i,moved)
            else:
                return move(nums,i+1,moved)
        # nums = move(nums,0,0)
        j = 0
        for i in range(len(nums)):
            if nums[i] != 0:
                temp = nums[i]
                nums[i] = nums[j]
                nums[j] = temp
                j += 1

    '''
        452. 用最少数量的箭引爆气球
            在二维空间中有许多球形的气球。对于每个气球，提供的输入是水平方向上，气球直径的开始和结束坐标。
            由于它是水平的，所以纵坐标并不重要，因此只要知道开始和结束的横坐标就足够了。开始坐标总是小于结束坐标。
            一支弓箭可以沿着 x 轴从不同点完全垂直地射出。在坐标 x 处射出一支箭，若有一个气球的直径的开始和结束坐标为 xstart，xend， 且满足  xstart ≤ x ≤ xend，则该气球会被引爆。可以射出的弓箭的数量没有限制。 弓箭一旦被射出之后，可以无限地前进。我们想找到使得所有气球全部被引爆，所需的弓箭的最小数量。
            给你一个数组 points ，其中 points [i] = [xstart,xend] ，返回引爆所有气球所必须射出的最小弓箭数。
            示例 1：
            输入：points = [[10,16],[2,8],[1,6],[7,12]]
            输出：2
            解释：对于该样例，x = 6 可以射爆 [2,8],[1,6] 两个气球，以及 x = 11 射爆另外两个气球
    '''
    def findMinArrowShots(self, points):
        #求并集，复杂度排序+n*n ,超时
        if 0:
            def point2list(point):
                return [i for i in  range(point[0] , point[1] + 1)]
            points.sort(key = lambda x:[x[0],x[1]])
            print(points)
            needed = 0
            set_need = set(point2list(points[0]))
            for i in range(1 , len(points)):
                set_need&=set(point2list(points[i]))
                print(i,set_need)
                if len(set_need) == 0:
                    needed += 1
                    set_need = set(point2list(points[i]))
            print(needed + 1)
            return needed+1
        
        #判断首尾
        needed = 1
        if len(points) == 1:
            return needed
        elif len(points) == 0:
            return 0
        points.sort(key = lambda x:[x[0],x[1]])
        the_min , the_max = points[0][0],points[0][1]
        # print(points)
        # print(0,the_min , the_max)
        for i in range(1 , len(points)):
            # print(i,the_min , the_max)
            if points[i][0] > the_max:#不相交
                the_min , the_max = points[i][0],points[i][1]
                needed += 1
            else:
                the_min = max(the_min,points[i][0])
                the_max = min(the_max,points[i][1])
        # print(needed)
        return needed

        #判断一边，按右排序; 右永远大于设定值，如果左小于设定，那就相交；否则更新设定为现在的右，箭数加1
        if len(points)==0:
            return 0
        # points.sort() 速度慢
        sortedpoints = sorted(points,key=lambda x:x[1])
        n=1
        tmp = sortedpoints[0][1]
        for point in sortedpoints[1:]:
            if point[0]>tmp:
                tmp = point[1]
                n = n+1             
        return n

    '''
        493. 翻转对
        给定一个数组 nums ，如果 i < j 且 nums[i] > 2*nums[j] 我们就将 (i, j) 称作一个重要翻转对。
        你需要返回给定数组中的重要翻转对的数量。
        示例 1:
        输入: [1,3,2,3,1]
        输出: 2
        示例 2:
        输入: [2,4,3,5,1]
        输出: 3
    '''
    def reversePairs(self, nums):
        pass

    
    '''
    767. 重构字符串
    给定一个字符串S，检查是否能重新排布其中的字母，使得两相邻的字符不同。
    若可行，输出任意可行的结果。若不可行，返回空字符串。
    示例 1:
    输入: S = "aab"
    输出: "aba"
    '''
    def reorganizeString(self, S):
        from collections import defaultdict
        d = defaultdict(lambda: 0)
        for s in S:
            d[s] += 1
        listd = list(zip(d.keys(), d.values()))
        listd.sort(key = lambda x:x[1] , reverse = True)
        listd = [list(i) for i in listd]
        if listd[0][1] - 1 > len(S) - listd[0][1]:return '' #不能插空
        result = ''
        while len(listd) - 1 > 0:
            for i in range(len(listd)):
                result += (listd[i][0])
                listd[i][1] -= 1
                if listd[i][1]  <  1:
                    del listd[i]
                    break
        
        if len(listd) == 0:return result
        #只剩下最多的，再插一次
        result = list(result)
        result.append('.')
        i = 0
        while  listd[0][1]:
            i +=1
            if result[i] != listd[0][0] and result[i+1] != listd[0][0]:
                result.insert(i+1,listd[0][0])
                listd[0][1] -= 1
                i += 1
        rr = ''
        for r in result:
            rr += r
        return rr[:-1]

    '''
    204. 计数质数
    统计所有小于非负整数 n 的质数的数量。
    '''
    def countPrimes(self, n):
        if n < 2:
            return 0

        num_list = [True]*n
        num_list[0], num_list[1] = False, False

        for i in range(2, int(pow(n, 0.5)) + 1):
            if num_list[i]:  # 如果i为质数(不是任何质数的倍数)
                num_list[i * i::i] = [False] * ((n - i * i - 1) // i + 1)  # 因为要包含i*i所以需要+1；因为n不在列表里，所以需要-1

        return sum(num_list)  # True就是1，False就是0，可以直接统计
    
    '''
    118. 杨辉三角
        给定一个非负整数 numRows，生成杨辉三角的前 numRows 行。
    '''
    def generate(self, numRows) :
        if numRows == 0:return []
        elif numRows == 1:return[[1]]

        results = [[1]]
        for i in range(1,numRows):
            results.append([results[i-1][j]+results[i-1][j+1]  for j in range(len(results[i - 1]) -1) ] )
            results[i].append(1)
            results[i].insert(0,1)
        print(results)
        return results

    '''
    738. 单调递增的数字
        给定一个非负整数 N，找出小于或等于 N 的最大的整数，同时这个整数需要满足其各个位数上的数字是单调递增。
        （当且仅当每个相邻位数上的数字 x 和 y 满足 x <= y 时，我们称这个整数是单调递增的。）
    '''
    def monotoneIncreasingDigits(self, N):
        if N < 10:return N
        list_num = [int(n) for n in list(str(N))]
        for i in range(1,len(list_num)):
            if list_num[i] >= list_num[i-1]:
                pass
            else:
                #溯源到其相等的第一个数
                while True:
                    if i - 1 == 0:break
                    if list_num[i-1] == list_num[i -2]:
                        i-=1
                    else:
                        break
                list_num[i-1] -= 1
                for j in range(i,len(list_num)):
                    list_num[j] = 9
        N = 0
        for i,n in  enumerate(list_num):N += n*10**(len(list_num) - i - 1)
        print(N)
        return N

    '''
    34. 在排序数组中查找元素的第一个和最后一个位置
        给定一个按照升序排列的整数数组 nums，和一个目标值 target。找出给定目标值在数组中的开始位置和结束位置。
        如果数组中不存在目标值 target，返回 [-1, -1]。

        进阶：  你可以设计并实现时间复杂度为 O(log n) 的算法解决此问题吗？
    '''
    def searchRange(self, nums, target):
        pass
        

    '''
        514. 自由之路
        您需要顺时针或逆时针旋转 ring 以使 key 的一个字符在 12:00 方向对齐，然后按下中心按钮，以此逐个拼写完 key 中的所有字符。
        旋转 ring 拼出 key 字符 key[i] 的阶段中：
            您可以将 ring 顺时针或逆时针旋转一个位置，计为1步。旋转的最终目的是将字符串 ring 的一个字符与 12:00 方向对齐，并且这个字符必须等于字符 key[i] 。
            如果字符 key[i] 已经对齐到12:00方向，您需要按下中心按钮进行拼写，这也将算作 1 步。按完之后，您可以开始拼写 key 的下一个字符（下一阶段）, 直至完成所有拼写。

        示例：
        输入: ring = "godding", key = "gd"
        输出: 4
        解释:
        对于 key 的第一个字符 'g'，已经在正确的位置, 我们只需要1步来拼写这个字符。 
        对于 key 的第二个字符 'd'，我们需要逆时针旋转 ring "godding" 2步使它变成 "ddinggo"。
        当然, 我们还需要1步进行拼写。
        因此最终的输出是 4。
    '''
    def findRotateSteps(self, ring, key):
        from collections import defaultdict as d
        import numpy as np
        dictKey = d()
        for k in key:
            if k in dictKey:#剔除一样的
                continue
            else:
                dictKey[k] = []
                for j,r in enumerate(ring):
                    if k == r:
                        dictKey[k].append(j)
        print(dictKey)

        lenth = len(ring)
        result = 0

        def updateDict(nums):
            for key in dictKey.keys():
                for i in range(len(dictKey[key])):
                    dictKey[key][i] += nums
                    if dictKey[key][i] < 0:
                        dictKey[key][i] += lenth
                    elif dictKey[key][i] > lenth -1:
                        dictKey[key][i] -= lenth

        for k in key:
            steps = [(lenth - lc) for  lc in dictKey[k] ]  +   dictKey[k]
            minStep =np.min(steps)
            index = steps.index(minStep)
            print("index",index,steps)
            if index < len(dictKey[k]):#顺时针,注意顺时针转过去是 lenth-取序号
                updateDict(minStep)
                result += minStep
                print('%s:顺时针:%d'%(k,minStep))
            else:#逆时针
                updateDict(-minStep)
                result += minStep
                print('%s:逆时针:%d'%(k,minStep))
            result += 1
            print(dictKey)

        print(result)
        return result
    


if __name__ == "__main__":
    S = Solution()
    # S.findRotateSteps(ring = "godding", key = "godding")
    
    # k = [[[10,16],[2,8],[1,6],[7,12]], [[1,2],[3,4],[5,6],[7,8]] ,  [[1,2]] ,  [[2,3],[2,3]]]
    # S.findMinArrowShots(k[2])
    # S.reversePairs([1,10,3,10,3,1])
    S.monotoneIncreasingDigits(123444345)
