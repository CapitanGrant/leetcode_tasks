import operator
import timeit
from abc import ABC, abstractmethod
from functools import reduce
from heapq import merge
from itertools import count
from typing import List, Optional, NamedTuple
from collections import Counter


class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution:
    def addBinary(self, a: str, b: str) -> str:
        num = 0
        lst = []
        max_len = max(len(a), len(b))
        a = a.zfill(max_len)
        b = b.zfill(max_len)
        for i in range(max_len - 1, -1, -1):
            sum = num
            sum += 1 if a[i] == '1' else 0
            sum += 1 if b[i] == '1' else 0
            lst.insert(0, '1' if sum % 2 else '0')
            num = 1 if sum > 1 else 0
        if num:
            lst.insert(0, '1')

        return str(''.join(lst))

    def search(self, nums: List[int], target: int) -> int:
        low = 0
        high = len(nums) - 1
        while low <= high:
            mid = (low + high) // 2
            guess = nums[mid]
            if guess == target:
                return mid
            if guess > target:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def longestCommonPrefix(self, strs: List[str]) -> str:
        if not strs:
            return ""

        min_len = len(min(strs, key=len))
        lcp = ""

        for i in range(min_len):
            char = strs[0][i]
            for j in strs:
                if i >= len(j) or j[i] != char:
                    return lcp
            lcp += char

        return lcp

    def isValid(self, s: str) -> bool:
        pass

    def mySqrt(self, x: int) -> int:
        low = 0
        high = x
        while low <= high:
            mid = (low + high) // 2
            if mid ** 2 <= x <= (mid + 1) ** 2:
                return mid
            if mid ** 2 > x:
                high = mid - 1
            else:
                low = mid + 1
        return -1

    def missingNumber(self, nums: List[int]) -> int:
        sum_num = sum(nums)
        len_num = len(nums)
        res_num = sum([i for i in range(0, len_num + 1)])
        return res_num - sum_num

    def intersection(self, nums1: List[int], nums2: List[int]) -> List[int]:
        nums1 = set(nums1)
        nums2 = set(nums2)
        lst = list(nums1 & nums2)
        return lst

    def isBadVersion(self, version: int, bad: int) -> bool:
        if version == bad:
            return True
        return False

    def firstBadVersion(self, n: int) -> int:
        low = 0
        high = n - 1
        while low < high:
            mid = (low + high) // 2
            if self.sBadVersion(mid):
                high = mid
            else:
                low = mid + 1
        return low

    def removeOccurrences(self, s: str, part: str) -> str:
        while part in s:
            if s.find(part) != -1:
                s = s[:s.find(part)] + s[s.find(part) + len(part):]
        return s

    def for_cicle(self, num: int) -> bool:
        s = str(num)
        result = []
        if s[0] == '-':
            result.append(int(s[0] + s[1]))
            for digit in s[2:]:
                result.append(int(digit))
        else:
            for digit in s:
                result.append(int(digit))
        nums = [i for i in result]
        lst = []
        for i in range(1, len(nums)):
            if nums[i] == nums[-i - 1]:
                lst.append(True)
            else:
                return False
        return all(lst)

    def mergeTwoLists(self, list1: Optional[ListNode], list2: Optional[ListNode]) -> Optional[ListNode]:
        return list(merge(list1, list2))

    def getConcatenation(self, nums: List[int]) -> List[int]:
        return nums * 2

    def minOperations(self, boxes: str) -> List[int]:
        n = len(boxes)
        lst = ['0'] * n
        for i in range(n):
            total = 0
            for j in range(n):
                if boxes[j] == '1':
                    total += abs(i - j)
            lst[i] = total
        return lst

    def finalValueAfterOperations(self, operations: List[str]) -> int:
        counter = 0
        for operation in operations:
            if '+' in operation:
                counter += 1
            elif '-' in operation:
                counter -= 1
        return counter

    def numIdenticalPairs(self, nums: List[int]) -> int:
        lst = []
        for i in range(len(nums)):
            for j in range(i + 1, len(nums)):
                if nums[i] == nums[j] and i < j:
                    lst.append((i, j))
        return len(lst)

    def getSneakyNumbers(self, nums: List[int]) -> List[int]:
        res = []
        lst = []
        for i in nums:
            if i not in lst:
                lst.append(i)
            else:
                res.append(i)
        return res

    def minimumOperations(self, nums: List[int]) -> int:
        count = 0
        for i in nums:
            if i % 3 != 0:
                count += 1
        return count

    def findArray(self, pref: List[int]) -> List[int]:
        result = [0] * len(pref)
        result[0] = pref[0]
        for i in range(1, len(pref)):
            result[i] = pref[i] ^ pref[i - 1]
        return result

    def shuffle(self, nums: List[int], n: int) -> List[int]:
        y = nums[n:]
        x = nums[:n]
        res = []
        for i in range(len(y)):
            res.append(x[i])
            res.append(y[i])
        return res

    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        merged = nums1 + nums2
        merged.sort()
        n = len(merged)
        if n % 2 == 0:
            return (merged[n // 2 - 1] + merged[n // 2]) / 2
        else:
            return merged[n // 2]

    def findThePrefixCommonArray(self, A: List[int], B: List[int]) -> List[int]:
        res = []
        for j in range(1, len(B) + 1):
            len_set = len(set(A[:j]) & set(B[:j]))
            res.append(len_set)
        return res

    def groupThePeople(self, groupSizes: List[int]) -> List[List[int]]:
        result = []
        groups = {}

        for ind, group_size in enumerate(groupSizes):
            if group_size not in groups:
                groups[group_size] = []

            groups[group_size].append(ind)
            if len(groups[group_size]) == group_size:
                result.append(groups[group_size])
                groups[group_size] = []

        return result

    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        max_num = max(candies)
        result = []
        for candidate in candies:
            if candidate + extraCandies >= max_num:
                result.append(True)
            result.append(False)
        return result

    def subsetXORSum(self, nums: List[int]) -> int:
        power_set = [[]]
        result = []
        for x in nums:
            for i in range(len(power_set)):
                tmp_list = power_set[i].copy()
                tmp_list.append(x)
                power_set.append(tmp_list)
        for i in range(1, len(power_set)):
            if len(power_set[i]) > 1:
                xor_sum = reduce(operator.xor, power_set[i])
                result.append(xor_sum)
            elif len(power_set[i]) == 1:
                result.append(power_set[i][0])
        return sum(result)

    def findWordsContaining(self, words: List[str], x: str) -> List[int]:
        res = [en for en, i in enumerate(words) if x in i]
        return res

    def numArmstrong(self, num: int) -> bool:
        sum = 0
        len_num = len(str(num))
        for i in str(num):
            digit = int(i)
            sum += digit ** len_num
        return num == sum

    def countConsistentStrings(self, allowed: str, words: List[str]) -> int:
        count = 0
        set_allowed = set(allowed)
        for word in words:
            flag = True
            for char in word:
                if char not in set_allowed:
                    flag = False
                    break
            if flag:
                count += 1
        return count

    def numberOfEmployeesWhoMetTarget(self, hours: List[int], target: int) -> int:
        count = 0
        for i in hours:
            if i >= target:
                count = count + 1
        return count

    def getFinalState(self, nums: List[int], k: int, multiplier: int) -> List[int]:
        for i in range(k):
            min_x = min(nums)
            ind = nums.index(min_x)
            nums[ind] = min_x * multiplier
        return nums

    def countPairs(self, nums: List[int], target: int) -> int:
        count_num = len(nums)
        first_ind = 0
        count = 0
        while count_num > 0:
            for i in range(first_ind + 1, len(nums)):
                if nums[first_ind] + nums[i] < target:
                    count = count + 1
            first_ind += 1
            count_num -= 1
        return count

    def minMovesToSeat(self, seats: List[int], students: List[int]) -> int:
        seats.sort()
        students.sort()
        count = 0
        for x, y in zip(seats, students):
            count += abs(x - y)

        return count

    def leftRightDifference(self, nums: List[int]) -> List[int]:
        result = []
        for i in range(len(nums)):
            leftSum = sum(nums[:i])
            rightSum = sum(nums[i + 1:])
            result.append(abs(leftSum - rightSum))
        return result

    def stableMountains(self, height: List[int], threshold: int) -> List[int]:
        result = []
        for i in range(1, len(height)):
            if height[i - 1] > threshold:
                result.append(i)
        return result

    def smallerNumbersThanCurrent(self, nums: List[int]) -> List[int]:
        temp = sorted(nums)
        d = {}
        for i, num in enumerate(temp):
            if num not in d:
                d[num] = i

        res = []
        for num in nums:
            res.append(d[num])

        return res

    def findMatrix(self, nums: List[int]) -> List[List[int]]:
        cnt = Counter(nums)
        result = []
        while sum(cnt.values()) > 0:
            row = []
            for num in list(cnt):
                if cnt[num] > 0:
                    row.append(num)
                    cnt[num] -= 1
            result.append(row)
        return result

    def mostWordsFound(self, sentences: List[str]) -> int:
        count = 0
        for i in sentences:
            words = len(i.split(' '))
            if count < words:
                count = words
        return count

    def minOperations(self, nums: List[int], k: int) -> int:
        count = 0
        for i in nums:
            if i < k:
                count += 1
        return count

    def numberOfPairs(self, nums1: List[int], nums2: List[int], k: int) -> int:
        count = 0
        for i in range(len(nums1)):
            for j in range(len(nums2)):
                product = nums2[j] * k
                if product > nums1[i]:
                    continue
                if nums1[i] % product == 0:
                    count += 1
        return count

    def createTargetArray(self, nums: List[int], index: List[int]) -> List[int]:
        res = []
        for i in range(len(nums)):
            res.insert(index[i], nums[i])
        return res

    def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
        result = 0
        for i in range(len(nums)):
            num = bin(i).count('1')
            if num == k:
                result += nums[i]
        return result

    def truncateSentence(self, s: str, k: int) -> str:

        result = s.split(' ')
        return ' '.join(result[:k])

    def arrayStringsAreEqual(self, word1: List[str], word2: List[str]) -> bool:
        x1 = ''.join(word1)
        x2 = ''.join(word2)
        return True if x1 == x2 else False

    def restoreString(self, s: str, indices: List[int]) -> str:
        res = {}
        result = []
        for x, y in zip(indices, s):
            res[x] = y
        for i in range(len(indices)):
            result.append(res[i])
        return ''.join(result)

    def countMatches(self, items: List[List[str]], ruleKey: str, ruleValue: str) -> int:
        if ruleKey == "type":
            index = 0
        elif ruleKey == "color":
            index = 1
        elif ruleKey == "name":
            index = 2
        else:
            return 0
        count = 0
        for item in items:
            if item[index] == ruleValue:
                count += 1

        return count

    def garbageCollection(self, garbage: List[str], travel: List[int]) -> int:
        count = 0
        glass = 0
        metal = 0
        pepper = 0
        for item in garbage:
            if "G" in item:
                glass += 1


        print(glass)

    def removeDuplicates(self, nums: List[int]) -> int:
        if not nums:
            return 0
        i = 0
        for j in range(1, len(nums)):
            if nums[j] != nums[i]:
                i += 1
                nums[i] = nums[j]
        return i + 1

    def transformArray(self, nums: List[int]) -> List[int]:
        res = []
        for i in nums:
            if i % 2 == 0:
                res.append(0)
            else:
                res.append(1)
        res.sort()
        return res

    def pivotArray(self, nums: List[int], pivot: int) -> List[int]:
        left = []
        right = []
        equal = []
        for num in nums:
            if num < pivot:
                left.append(num)
            elif num > pivot:
                right.append(num)
            else:
                equal.append(num)
        return left + equal + right

    def subarraySum(self, nums: List[int]) -> int:
        res = 0
        for i in range(len(nums)):
            start = max(0, i - nums[i])
            num = nums[start : i+1]
            subarrey_sum = sum(num)
            res += subarrey_sum
        return res
        
    def sortPeople(self, names: List[str], heights: List[int]) -> List[str]:
        lst = []
        for i in zip(names, heights):
            lst.append(i)
        res = sorted(lst, key=lambda x: x[1], reverse=True)
        result = [i[0] for i in res]
        return result

    def leftRightDifference(self, nums: List[int]) -> List[int]:
        n = len(nums)
        left_sum = [0] * n
        right_sum = [0] * n
        answer = [0] * n


        for i in range(1, n):
            left_sum[i] = left_sum[i - 1] + nums[i - 1]


        for i in range(n - 2, -1, -1):
            right_sum[i] = right_sum[i + 1] + nums[i + 1]


        for i in range(n):
            answer[i] = abs(left_sum[i] - right_sum[i])

        return answer
    
    
    def numberGame(self, nums: List[int]) -> List[int]:
        res = []
        nums.sort()
        for i in range(0, len(nums), 2):
            if i + 1 < len(nums):
                res.append(nums[i+1])
                res.append(nums[i])
        return res
        
    def minimumAverage(self, nums: List[int]) -> float:
        res = []
        nums_sorted = sorted(nums)
        while len(nums_sorted) > 0:
            min_element = nums_sorted.pop(0)
            max_element = nums_sorted.pop()
            average = (min_element + max_element) / 2
            res.append(average)
        return min(res)

    def prefixCount(self, words: List[str], pref: str) -> int:
        res = []
        for i in words:
            stop = len(pref)
            if i[0:stop] == pref:
                res.append(i)
        return len(res)

    def firstPalindrome(self, words: List[str]) -> str:
        for word in words:
            if word == word[::-1]:
                return word
        return ""

    def scoreOfString(self, s: str) -> int:
        res = 0
        for i in range(len(s)):
            if i + 1 < len(s):
                res += abs(ord(s[i]) - ord(s[i + 1]))
        return res

    def theMaximumAchievableX(self, num: int, t: int) -> int:
        return num + 2 * t

    
    def convertTemperature(self, celsius: float) -> List[float]:
        Kelvin = celsius + 273.15
        Fahrenheit = celsius * 1.80 + 32.00
        return [Kelvin, Fahrenheit]

    
    def defangIPaddr(self, address: str) -> str:
        return address.replace('.', '[.]')

    
    def numJewelsInStones(self, jewels: str, stones: str) -> int:
        count = 0
        set_count = set(jewels)
        for i in stones:
            if i in set_count:
                count += 1
        return count

    def minPartitions(self, n: str) -> int:
        return max(int(i) for i in n)


    def convertDateToBinary(self, date: str) -> str:
        date_s = date.split('-')
        res = []
        for i in date_s:
            _ = bin(int(i))[2:]
            res.append(_)
        return "-".join(res)
    
    
    def threeConsecutiveOdds(self, arr: List[int]) -> bool:
        count = 0
        for i in arr:
            if i % 2 != 0:
                count += 1
                if count == 3:
                    return True
            else:
                count = 0
        return False
    
    def smallestEvenMultiple(self, n: int) -> int:
        if n % 2 == 0:
            return n
        else:
            return n * 2


    def isStrictlyPalindromic(self, n: int) -> bool:
        return False



sol = Solution()

print(sol.removeDuplicates(nums = [0,0,1,1,1,2,2,3,3,4]))


class Coordinate(NamedTuple):
    lat: float
    lon: float

    def __str__(self):
        ns = 'N' if self.lat >= 0 else 'S'
        we = 'E' if self.lon >= 0 else 'W'
        return f'{abs(self.lat):.1f}°{ns}, {abs(self.lon):.1f}°{we}'

# setup_code = """
# from typing import List
# from collections import Counter
# class Solution:
#     def sumIndicesWithKSetBits(self, nums: List[int], k: int) -> int:
#         result = 0
#         for i in range(len(nums)):
#             num = Counter(bin(i)[2:])
#             if num['1'] == k:
#                 result += nums[i]
#         return result
# """
#
# # Код для тестирования
# test_code = """
# def main():
#     sol = Solution()
#     sol.sumIndicesWithKSetBits(nums = [5,10,1,5,2], k = 1)
# main()
# """
#
# # Измерение времени выполнения
# elapsed_time = timeit.timeit(test_code, setup=setup_code, number=100) / 100
# print('Elapsed time:', elapsed_time)
#
# # Input: groupSizes = [3,3,3,3,3,1,3]
# # Output: [[5],[0,1,2],[3,4,6]]
# # Input: groupSizes = [2,1,3,3,3,2]
# # Output: [[1],[0,5],[2,3,4]]
# # Elapsed time: 1.0269999620504676e-06
