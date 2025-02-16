import functools
import timeit
from heapq import merge
from typing import List, Optional


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

sol = Solution()
print(sol.findThePrefixCommonArray(A=[2, 3, 1], B=[3, 1, 2]))

# setup_code = """
# from typing import List
#
# class Solution:
#     def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
#         merged = nums1 + nums2
#         merged.sort()
#         n = len(merged)
#         if n % 2 == 0:
#             return (merged[n // 2 - 1] + merged[n // 2]) / 2
#         else:
#             return merged[n // 2]
# """
#
# # Код для тестирования
# test_code = """
# def main():
#     sol = Solution()
#     sol.findMedianSortedArrays(nums1=[1, 2, 3, 4, 5], nums2=[6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17])
# main()
# """
#
# # Измерение времени выполнения
# elapsed_time = timeit.timeit(test_code, setup=setup_code, number=100) / 100
# print('Elapsed time:', elapsed_time)


# Elapsed time: 1.0269999620504676e-06
# print(sol.isValid(s="()")) # true
# print(sol.isValid(s="()[]{}")) # true
# print(sol.isValid(s="(]")) # false
#
#
# print(sol.longestCommonPrefix(strs=["flower", "flow", "flight"]))
# print(sol.longestCommonPrefix(strs=["dog","racecar","car"]))
#
#
# print(sol.addBinary('000100', '110010'))  # 110110
# print(sol.addBinary('1010', '1011'))  # 10101
